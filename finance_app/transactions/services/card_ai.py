"""Structured vision fallback for scanned card statements, using the configured PDF model."""
import base64
import io
import json
import os
from datetime import date
from decimal import Decimal
from pathlib import Path


def parse_card_images(path,passwords=()):
    from openai import OpenAI
    from pdf2image import convert_from_path
    key=os.getenv('OPENAI_API_KEY','')
    if not key:raise ValueError('Este PDF necesita lectura de imágenes. Configura OpenAI o carga una versión con texto.')
    pages=None
    for password in ['',*passwords]:
        try:pages=convert_from_path(path,dpi=200,userpw=password);break
        except Exception:continue
    if not pages:raise ValueError('No se pudieron abrir las páginas del PDF; revisa su contraseña.')
    nullable_number={'type':['number','null']};nullable_text={'type':['string','null']}
    row_fields={
        'date':{'type':'string'},'description':{'type':'string'},'reference':{'type':'string'},
        'amount':{'type':'number'},'kind':{'type':'string','enum':['purchase','refund','payment','interest','fee','tax','advance','adjustment']},
        'billed_amount':nullable_number,'remaining_balance':nullable_number,
        'installment':{'type':['integer','null']},'installments':{'type':['integer','null']},
    }
    fields={'bank':nullable_text,'last4':nullable_text,'period_start':nullable_text,'period_end':nullable_text,
            'currency':nullable_text,'opening_balance':nullable_number,'closing_balance':nullable_number,
            'purchases_total':nullable_number,'payments_total':nullable_number,
            'rows':{'type':'array','items':{'type':'object','properties':row_fields,'required':list(row_fields),'additionalProperties':False}}}
    schema={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
    metadata={};rows=[]
    client=OpenAI(api_key=key,timeout=180)
    for number,page in enumerate(pages,1):
        buffer=io.BytesIO();page.convert('RGB').save(buffer,format='JPEG',quality=90)
        prompt='''Extract CREDIT CARD statement data from this page, not savings account rows.
Return every card transaction, including payments, refunds, interest, insurance, fees and prior-month installment purchases.
amount is the ORIGINAL PURCHASE PRINCIPAL in the statement currency, not the installment or remaining balance.
Charges and purchases are positive; payments and refunds negative. A reversed payment is a positive adjustment, not a payment.
A zero amount-to-pay does not mean the original purchase was zero: it may already have been paid.
Do not count summary totals as transactions. Exception: an interest/fee charge present ONLY in the summary must be returned once on the closing date.
Keep original descriptions, authorization references and original purchase dates. Do not append original USD values to a COP amount.
Return bank, last4, period dates (ISO), currency and summary balances when visible; otherwise null.
Bank names must be Davivienda or Banco de Bogotá when applicable. Never guess missing amounts.
Context from previous pages: '''+json.dumps(metadata,default=str)
        result=client.responses.create(model=os.getenv('OPENAI_PDF_MODEL','gpt-6-luna'),store=False,
            input=[{'role':'user','content':[{'type':'input_text','text':prompt},{'type':'input_image','image_url':'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()}]}],
            text={'format':{'type':'json_schema','name':'card_statement','strict':True,'schema':schema}})
        if result.status!='completed':raise ValueError('La lectura de la tarjeta quedó incompleta; no se importó.')
        data=json.loads(result.output_text)
        for key,value in data.items():
            if key!='rows' and value is not None:
                if key in {'bank','last4','period_start','period_end','currency'} and key in metadata and metadata[key]!=value:
                    raise ValueError('El PDF contiene más de una tarjeta o período. Carga cada extracto por separado.')
                metadata.setdefault(key,value)
        for row in data['rows']:
            row['date']=date.fromisoformat(row['date']);row['source_page']=number
            for field in ['amount','billed_amount','remaining_balance']:
                if row[field] is not None:
                    value=Decimal(str(row[field]))
                    if not value.is_finite() or abs(value)>=Decimal('100000000000000'):raise ValueError('Importe de tarjeta inválido.')
                    row[field]=value.quantize(Decimal('.01'))
            if row['kind'] in {'payment','refund'} and row['amount']>0:raise ValueError('Signo incorrecto en abono o devolución.')
            rows.append(row)
    if not all(metadata.get(k) for k in ['bank','last4','period_start','period_end','currency']):raise ValueError('No se identificó la tarjeta o su período completo.')
    if not str(metadata['last4']).isdigit() or len(str(metadata['last4']))!=4:raise ValueError('Número de tarjeta inválido.')
    for field in ['period_start','period_end']:metadata[field]=date.fromisoformat(metadata[field])
    if metadata['period_start']>metadata['period_end']:raise ValueError('Período inválido.')
    for field in ['opening_balance','closing_balance','purchases_total','payments_total']:
        metadata[field]=Decimal(str(metadata[field])).quantize(Decimal('.01')) if metadata.get(field) is not None else None
    metadata['rows']=rows;metadata['validation_notes']=['Extracto leído con IA a partir de imágenes. Comprueba las filas y saldos con el PDF original.']
    paid=sum((-r['amount'] for r in rows if r['kind']=='payment'),Decimal(0))
    if metadata['payments_total'] is not None and abs(paid-metadata['payments_total'])>Decimal(2):metadata['validation_notes'].append(f"Abonos identificados {paid}; resumen {metadata['payments_total']}. Requiere revisión.")
    return metadata
