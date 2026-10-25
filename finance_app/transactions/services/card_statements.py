"""Credit card ledger. Purchases are stored once, independently of bank cash flows."""
import hashlib
import re
from calendar import monthrange
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.db import transaction

from finance_app.core.services.classification import normalize,is_housing_loan,is_education_payment
from finance_app.transactions.models import CardMovement, CardStatement, CardStatementLine, CreditCard

ZERO = Decimal('0.00')
MONTHS = {'ene':1,'jan':1,'feb':2,'mar':3,'abr':4,'apr':4,'may':5,'jun':6,'jul':7,'ago':8,'aug':8,'sep':9,'oct':10,'nov':11,'dic':12,'dec':12}
NUM = r'-?\d[\d,]*(?:\.\d{1,2})?'


def cash(text):
    return Decimal(text.replace(',', '').replace('$','').strip()).quantize(Decimal('.01'))


def compact_date(text):
    m = re.fullmatch(r'(\d{1,2})[ /.-]*([A-Za-z]{3})[ /.-]*(\d{2,4})', text.strip(), re.I)
    if not m: raise ValueError(f'Fecha de tarjeta no reconocida: {text}')
    year=int(m[3]);year=year+2000 if year<100 else year
    return date(year, MONTHS[m[2].lower()], int(m[1]))


def household_category(description, kind='purchase'):
    """Owner-requested rules take precedence over broad commerce/processor labels."""
    if kind not in {'purchase', 'refund'}:
        return None
    text=normalize(description)
    if is_education_payment(description):
        return 'educacion'
    if re.search(r'\b(?:droguerias?|drogarias?|drogas|drog|farmacias?|farmatodo|farma)\b',text):
        return 'salud'
    if re.search(r'\boxxo(?:\b|\d)',text):
        return 'mercado'
    if is_housing_loan(description):
        return 'hogar'
    if re.search(r'\b(?:homecenter|homesentry|ikea|tugo|ferreterias?|muebles|colchones|menaje|electrodomesticos|arriendo)\b|home sentry|gastos? (?:de (?:la )?)?casa|articulos (?:de|para el) hogar|decoracion|credito (?:de |para )?vivienda|credito hipotecario|prestamo hipotecario|cuota hipotecaria|leasing habitacional|pago.*hipoteca|administracion (?:de )?(?:conjunto|edificio|vivienda)',text):
        return 'hogar'
    provider=re.search(r'\b(?:comcel|telmex|movistar|claro|tigo|wom|etb|enel|codensa|vanti|eaab|acueducto|alcantarillado)\b',text)
    utility=re.search(r'servicios? publicos?|gas natural|energia electrica|telefonia|plan(?:es)? de datos|paquete(?:s)? de datos|datos moviles|recarga.*(?:celular|datos)|pago.*celular',text)
    internet='internet' in text and not re.search(r'transferencia|\bach\b|comision|servilinea|banca movil',text)
    if provider or utility or internet:
        return 'servicios'
    return None


def merchant_category(description, kind='purchase'):
    text=normalize(description)
    if kind in {'interest','fee','tax'}:return 'Costos financieros','financieros'
    if kind=='payment':return 'Pago de tarjeta','financieros'
    category=household_category(description,kind)
    if category:
        return description.strip()[:255],category
    rules=[
        ('Apple','tecnologia',r'apple\.com|itunes|apple store'),
        ('Mercado Libre','compras',r'mercado\s*libre|mercadolibre|meli\*'),
        ('Amazon','compras',r'amazon|amzn'),
        ('Microsoft','tecnologia',r'microsoft|msft'),
        ('OpenAI','tecnologia',r'openai|chatgpt'),
        ('Netflix','entretenimiento',r'netflix'),
        ('Spotify','entretenimiento',r'spotify'),
        ('Steam','entretenimiento',r'steamgames|steampowered|steam purchase'),
        ('Cine Colombia','entretenimiento',r'cine colombia|cinemark'),
        ('DiDi Food','restaurantes',r'didi.*food'),
        ('DiDi','transporte',r'didi'),
        ('Uber Eats','restaurantes',r'uber.*eats'),
        ('Uber','transporte',r'uber'),
        ('Rappi','restaurantes',r'rappi'),
        ('Mercado Pago','compras',r'mercado\s*pago'),
        ('','restaurantes',r'restaur|crepes|waffles|taqueria|dominos|pizza|kfc|mcdonald|burger|starbucks|coffipan|tanuki|el corral|colmill|pub|sushi'),
        ('','mercado',r'supermerc|olaya|tienda d1|ara\b|exito|carulla|jumbo|carnes|pricesmart'),
        ('','salud',r'drogas|farmatodo|farmacia|medicina|eps\b|\bips\b|salud'),
        ('','transporte',r'parking|parqueadero|gasolina|terpel|primax|eds\b|cabify|peaje'),
        ('','viajes',r'hotel|avianca|latam|booking|airbnb|wingo|aerolinea'),
        ('','servicios',r'comcel|telmex|movistar|claro|acueducto|enel|gas natural'),
        ('','compras',r'falabella|wish|aliexpress|temu|arturo calle|zara|h&m|decathlon'),
    ]
    for merchant,category,pattern in rules:
        if re.search(pattern,text):return merchant or description.strip()[:255],category
    return description.strip()[:255], 'otros'


def movement_kind(description):
    text=normalize(description)
    if re.match(r'^pago(?:$|\s+(?:banca|por|tarjeta|pse|efectivo|debito|recibido))',text):return 'payment'
    if re.search(r'devoluc|reversion|reverso|anulacion|nota credito|^abono',text):return 'refund'
    if 'interes' in text:return 'interest'
    if re.search(r'cuota.*manejo|seguro|comision|otros cargos',text):return 'fee'
    if re.search(r'\biva\b|\bimpuesto\b|gravamen',text):return 'tax'
    if 'avance' in text:return 'advance'
    return 'purchase'


def _summary(text, pattern, dollar=True, last=False):
    matches=re.findall(pattern+r'\s*'+(r'\$\s*' if dollar else '')+'('+NUM+')',text,re.I)
    return cash(matches[-1] if last else matches[0]) if matches else None


def parse_card_pages(pages):
    text='\n'.join(pages)
    ntext=normalize(text)
    rows=[];notes=[]
    modern='detalle de movimientos del mes' in ntext
    dav='davivienda' in ntext or modern
    bank='Davivienda' if dav else 'Banco de Bogotá'
    if dav:
        card=re.search(r'(?:#\s*|credito\s*:\s*)?(\d{4})\s+(?:\d{4}|\*{4})\s+(?:\d{4}|\*{4})\s+(\d{4})',text,re.I)
        if not card:raise ValueError('No se identificó el número de la tarjeta.')
        last4=card[2]
        if modern:
            period=re.search(r'(\d{1,2}/[A-Za-z]{3}/\d{4})\s*-\s*(\d{1,2}/[A-Za-z]{3}/\d{4})',text)
            if not period:raise ValueError('No se identificó el período de facturación.')
            start,end=map(compact_date,period.groups())
        else:
            period=re.search(r'Periodo\s+liquidado\s+([A-Za-z]{3})\.?\s*(\d{1,2})\s*/(\d{2,4})\s*-\s*([A-Za-z]{3})\.?\s*(\d{1,2})\s*/(\d{2,4})',text,re.I)
            if not period:raise ValueError('No se identificó el período liquidado.')
            start=compact_date(f'{period[2]}{period[1]}{period[3]}');end=compact_date(f'{period[5]}{period[4]}{period[6]}')
        opening=_summary(text,r'Saldo (?:mes )?anterior')
        closing=_summary(text,r'Pago total' if modern else r'Saldo Total')
        favor=_summary(text,r'Saldo a [Ff]avor') or ZERO
        if closing is not None:closing-=favor
        purchases=_summary(text,r'\+Compras del mes' if modern else r'Consumos mes')
        payments=_summary(text,r'-Pagos y abonos' if modern else r'Pagos Mes')
        payment_section=False;other_section=False
        for page_no,page in enumerate(pages,1):
            for line in page.splitlines():
                norm=normalize(line)
                if 'detalle aplicacion de pagos' in norm:payment_section=True;other_section=False
                if 'detalle de movimientos del mes' in norm:payment_section=False
                if 'detalle otros cargos' in norm:other_section=True;payment_section=False
                if not modern:
                    m=re.match(r'^\s*(\d{4,})\s+(\d{8})\s+(.+?)\s+\$',line)
                    if not m:continue
                    when=date(int(m[2][:4]),int(m[2][4:6]),int(m[2][6:8]));desc=m[3].strip();ref=m[1]
                    values=re.findall(r'\$\s*('+NUM+r')\s*([+-]?)',line[m.end()-1:])
                    if len(values)<3:raise ValueError('Fila incompleta de tarjeta; necesita revisión.')
                    amount=cash(values[0][0]);kind=movement_kind(desc)
                    billed=cash(values[1][0])*(-1 if values[1][1]=='-' else 1)
                    if kind=='payment' and values[1][1]=='+':
                        kind='adjustment'
                    elif kind in {'payment','refund'}:amount=-abs(amount)
                    elif billed<0:kind='refund';amount=-abs(amount)
                    inst=re.search(r'(\d+)\s*/\s*(\d+)\s+[\d.]',line)
                    rows.append(dict(date=when,description=desc,reference=ref,amount=amount,kind=kind,
                                     billed_amount=billed,remaining_balance=cash(values[2][0]),
                                     installment=int(inst[1]) if inst else None,installments=int(inst[2]) if inst else None,source_page=page_no))
                else:
                    m=re.match(r'^\s*(\d{1,2}[A-Za-z]{3}\d{4})\s+(.+)',line)
                    if not m:continue
                    when=compact_date(m[1]);tail=m[2];values=re.findall(r'\$\s*('+NUM+')',tail)
                    if payment_section and values:
                        rows.append(dict(date=when,description='PAGO',reference='',amount=-abs(cash(values[0])),kind='payment',billed_amount=-abs(cash(values[0])),source_page=page_no));continue
                    inst=re.search(r'(\d+)\s+de\s+(\d+)',tail)
                    if inst and len(values)>=4:
                        desc=tail.split('$')[0].strip();kind=movement_kind(desc);amount=cash(values[0]);amount=-abs(amount) if kind in {'payment','refund'} else amount
                        after_money=re.split(r'\$\s*'+NUM,tail)[-1].split();ref=after_money[1] if len(after_money)>1 else ''
                        rows.append(dict(date=when,description=desc,reference=ref,amount=amount,kind=kind,billed_amount=cash(values[2]),remaining_balance=cash(values[3]),installment=int(inst[1]),installments=int(inst[2]),source_page=page_no))
                    elif other_section and len(values)==1:
                        before=tail.split('$')[0].strip();match=re.match(r'(.+?)\s+(\d{4,})$',before)
                        if match:rows.append(dict(date=when,description=match[1].strip(),reference=match[2],amount=cash(values[0]),kind=movement_kind(match[1]),billed_amount=cash(values[0]),source_page=page_no))
    else:
        card=re.search(r'Tarjeta\s+N[uú]mero\s+(\d{12,19})',text,re.I)
        if not card:raise ValueError('No se reconoce un extracto de tarjeta de crédito.')
        last4=card[1][-4:]
        end_match=re.search(r'Fecha\s+Facturaci[oó]n[^\n]*\n\s*(\d{2}/\d{2}/\d{4})',text,re.I)
        if not end_match:
            dates=re.findall(r'\b\d{2}/\d{2}/\d{4}\b',text[:4500]);end_raw=dates[0] if dates else ''
        else:end_raw=end_match[1]
        d,m,y=map(int,end_raw.split('/'));end=date(y,m,d)
        previous_month=12 if m==1 else m-1;previous_year=y-1 if m==1 else y
        start=date(previous_year,previous_month,min(d,monthrange(previous_year,previous_month)[1]))+timedelta(days=1)
        notes.append('El inicio del período se estimó a partir de la fecha de facturación; verifica las compras cercanas al corte.')
        for page_no,page in enumerate(pages,1):
            for line in page.splitlines():
                m=re.match(r'^\s*(\d{4,})\s+(.+?)\s+(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(\d{1,2})\s+(.+)$',line)
                if not m:continue
                d,mo,y=map(int,m[3].split('/'));when=date(y,mo,d);pd,pm,py=map(int,m[4].split('/'));processed=date(py,pm,pd);desc=m[2].strip();nums=m[6].split()
                if len(nums)<5:raise ValueError('Fila incompleta Banco de Bogotá.')
                amount=cash(nums[0]);kind=movement_kind(desc)
                if kind=='purchase' and int(m[5])==0 and Decimal(nums[1])==0:kind='refund'
                if kind in {'payment','refund'}:amount=-abs(amount)
                rows.append(dict(date=when,processed_date=processed,description=desc,reference=m[1],amount=amount,kind=kind,billed_amount=cash(nums[2]),remaining_balance=cash(nums[4]),installments=int(m[5]),source_page=page_no))
        opening=_summary(text,r'Saldo Anterior',False)
        closing=_summary(text,r'=\s*Pago Total',False)
        purchases=_summary(text,r'\+\s*Compras',False,True)
        payments=_summary(text,r'-\s*Pagos y Cr[eé]ditos',False)
        # Charges reported only in the summary are separate from principal purchases.
        for label,kind in [(r'Intereses Corrientes','interest'),(r'Intereses de Mora','interest'),(r'Otros cargos y d[eé]bitos','fee')]:
            charge=_summary(text,r'\+\s*'+label+r'\s*\(\d\)',False,True)
            if charge and charge>0:rows.append(dict(date=end,description=re.sub(r'\[.*?\]','',label).replace('\\',''),reference='summary-'+label,amount=charge,kind=kind,billed_amount=charge,source_page=1))
    if not rows and not (purchases == ZERO and payments == ZERO and opening == closing):
        raise ValueError('No se encontraron movimientos legibles en el extracto.')
    if start>end:raise ValueError('Período del extracto inválido.')
    paid=sum((-r['amount'] for r in rows if r['kind']=='payment'),ZERO)
    credited=sum((-r['amount'] for r in rows if r['kind']=='refund'),ZERO)
    if payments is not None and abs(payments-paid-(credited if bank=='Banco de Bogotá' else ZERO))>Decimal('2'):
        notes.append(f'Pagos del resumen {payments} y filas de pagos {paid} no coinciden; revisa abonos/devoluciones o filas faltantes.')
    purchased=sum((r['amount'] for r in rows if r['kind']=='purchase' and start<=r.get('processed_date',r['date'])<=end),ZERO)
    if purchases is not None and abs(purchases-purchased)>Decimal('2'):
        notes.append(f'Compras del resumen {purchases} y compras del período identificadas {purchased}; puede haber cargos de otro corte o filas por revisar.')
    return dict(bank=bank,last4=last4,period_start=start,period_end=end,currency='COP',opening_balance=opening,closing_balance=closing,purchases_total=purchases,payments_total=payments,validation_notes=notes,rows=rows)


def read_card_pdf(path,passwords=()):
    from pypdf import PdfReader
    reader=PdfReader(path)
    if reader.is_encrypted and not any(reader.decrypt(pw) for pw in ['',*passwords]):
        raise ValueError('No se pudo abrir el extracto. Revisa la contraseña.')
    pages=[p.extract_text(extraction_mode='layout') or '' for p in reader.pages]
    try:
        return parse_card_pages(pages)
    except ValueError:
        from finance_app.transactions.services.card_ai import parse_card_images
        return parse_card_images(path,passwords)


@transaction.atomic
def store_card_statement(user,path,parsed,refresh=False):
    content_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    card,_=CreditCard.objects.get_or_create(user=user,bank=parsed['bank'],last4=parsed['last4'])
    existing=CardStatement.objects.filter(card=card,period_start=parsed['period_start'],period_end=parsed['period_end'],currency=parsed['currency']).first()
    if existing and not refresh:return existing,False
    source=f'{user.pk}_{content_hash}.pdf'
    fields={k:v for k,v in parsed.items() if k not in {'rows','bank','last4'}}
    if existing:
        statement=existing
        for key,value in fields.items():setattr(statement,key,value)
        statement.file_hash=content_hash;statement.source_pdf=source;statement.save()
        statement.lines.all().delete()
    else:
        statement=CardStatement.objects.create(card=card,file_hash=content_hash,source_pdf=source,**fields)
    occurrences=Counter()
    for row in parsed['rows']:
        identity='|'.join([str(row['date']),row.get('reference',''),normalize(row['description']),str(cash(str(row['amount']))),parsed['currency']])
        occurrences[identity]+=1
        fingerprint=hashlib.sha256(f'{identity}|{occurrences[identity]}'.encode()).hexdigest()
        merchant,category=merchant_category(row['description'],row['kind'])
        movement,_=CardMovement.objects.get_or_create(card=card,fingerprint=fingerprint,defaults={
            'date':row['date'],'kind':row['kind'],'description':row['description'],'amount':row['amount'],
            'reference':row.get('reference',''),'currency':parsed['currency'],'merchant':merchant,'category':category})
        CardStatementLine.objects.create(statement=statement,movement=movement,**{k:row[k] for k in ['billed_amount','remaining_balance','installment','installments','source_page'] if k in row})
    if existing:
        CardMovement.objects.filter(card=card,statement_lines__isnull=True).delete()
    root=Path(settings.MEDIA_ROOT)/'cards';root.mkdir(parents=True,exist_ok=True)
    dest=root/source
    if not dest.exists():
        import shutil
        shutil.copyfile(path,dest)
    return statement,not bool(existing)
