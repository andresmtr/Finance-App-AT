from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q, Sum
from django.db.models.functions import TruncMonth
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme

from finance_app.transactions.models import CardMovement, CardPaymentAllocation, CardStatement, CardStatementLine, CardExpenseAllocation, CreditCard, Transaction
from finance_app.transactions.services.card_reconciliation import allocate_payment, auto_reconcile, bank_payments, bank_remaining, card_remaining, allocate_expenses, bank_expense_remaining, expense_remaining
from finance_app.transactions.services.card_statements import read_card_pdf, store_card_statement

EXPENSE_KINDS=['purchase','refund','interest','fee','tax']
ZERO=Decimal('0.00')


def _dates(month):
    try:
        year,number=map(int,month.split('-'));start=date(year,number,1)
        end=date(year+1,1,1) if number==12 else date(year,number+1,1)
        return start,end
    except (ValueError,TypeError):return None,None


def _sum(qs):return qs.aggregate(total=Sum('amount'))['total'] or ZERO


@login_required
def cards(request):
    cards=CreditCard.objects.filter(user=request.user).order_by('bank','last4')
    movements=CardMovement.objects.filter(card__user=request.user).select_related('card').prefetch_related('allocations','expense_allocations','statement_lines')
    statements=CardStatement.objects.filter(card__user=request.user).select_related('card')
    bank=bank_payments(request.user)
    month=request.GET.get('month','');selected=request.GET.get('card','');currency=request.GET.get('currency','COP')
    if currency not in {'COP','USD'}:currency='COP'
    movements=movements.filter(currency=currency);statements=statements.filter(currency=currency);bank=bank.filter(currency=currency)
    if selected.isdigit():
        movements=movements.filter(card_id=int(selected));statements=statements.filter(card_id=int(selected))
        bank=bank.filter(card_allocations__card_payment__card_id=int(selected)).distinct() | bank.filter(description__icontains=cards.filter(pk=int(selected)).values_list('last4',flat=True).first() or 'NO_CARD')
    start,end=_dates(month)
    if start:
        movements=movements.filter(date__gte=start,date__lt=end);statements=statements.filter(period_end__gte=start,period_end__lt=end);bank=bank.filter(date__gte=start,date__lt=end)
    expense=movements.filter(kind__in=EXPENSE_KINDS)
    category=request.GET.get('category','');merchant=request.GET.get('merchant','').strip()
    if category:expense=expense.filter(category=category)
    if merchant:expense=expense.filter(merchant__icontains=merchant)
    if request.GET.get('export')=='1':
        from finance_app.transactions.services.exports import workbook_response
        allocations=CardPaymentAllocation.objects.filter(bank_transaction__user=request.user).filter(
            Q(bank_transaction__in=bank) | Q(card_payment__in=movements.filter(kind='payment'))
        ).select_related('bank_transaction','card_payment__card').order_by('pk')
        sheets=[
            ('Consumos',['id','fecha_compra','banco','tarjeta','descripcion','comercio','categoria','tipo','valor_original','moneda','referencia'],
             [(m.pk,m.date,m.card.bank,m.card.last4,m.description,m.merchant,m.get_category_display(),m.get_kind_display(),m.amount,m.currency,m.reference) for m in expense.order_by('date','pk')]),
            ('Abonos tarjeta',['id','fecha','tarjeta','descripcion','abono','pendiente','moneda'],
             [(m.pk,m.date,str(m.card),m.description,abs(m.amount),card_remaining(m),m.currency) for m in movements.filter(kind='payment').order_by('date','pk')]),
            ('Pagos de cuenta',['id','fecha','banco','cuenta','descripcion','salida','pendiente','moneda'],
             [(t.pk,t.date,t.bank,t.account_last4,t.description,abs(t.amount),bank_remaining(t),t.currency) for t in bank.order_by('date','pk')]),
            ('Conciliaciones',['id','movimiento_cuenta_id','fecha_cuenta','abono_tarjeta_id','fecha_tarjeta','tarjeta','valor_cruzado','moneda','metodo','nota'],
             [(a.pk,a.bank_transaction_id,a.bank_transaction.date,a.card_payment_id,a.card_payment.date,str(a.card_payment.card),a.amount,a.bank_transaction.currency,a.method,a.notes) for a in allocations]),
            ('Gastos asociados',['pago_cuenta_id','fecha_pago','gasto_tarjeta_id','fecha_compra','comercio','valor_asociado','moneda','nota'],
             [(a.bank_transaction_id,a.bank_transaction.date,a.card_expense_id,a.card_expense.date,a.card_expense.merchant,a.amount,a.bank_transaction.currency,a.notes) for a in CardExpenseAllocation.objects.filter(bank_transaction__user=request.user).filter(Q(bank_transaction__in=bank)|Q(card_expense__in=expense)).select_related('bank_transaction','card_expense')]),
            ('Extractos',['id','tarjeta','inicio','corte','moneda','saldo_anterior','saldo_cierre','compras_resumen','pagos_resumen','revisado','observaciones'],
             [(st.pk,str(st.card),st.period_start,st.period_end,st.currency,st.opening_balance,st.closing_balance,st.purchases_total,st.payments_total,st.reviewed,'; '.join(st.validation_notes)) for st in statements]),
            ('Detalle extractos',['extracto_id','movimiento_id','fecha_compra','valor_original','valor_a_pagar','saldo_pendiente','cuota','cuotas','pagina_pdf'],
             [(line.statement_id,line.movement_id,line.movement.date,line.movement.amount,line.billed_amount,line.remaining_balance,line.installment,line.installments,line.source_page) for line in CardStatementLine.objects.filter(statement__in=statements).select_related('movement').order_by('statement_id','pk')]),
            ('Alcance',['campo','valor'], [('mes',month or 'Todos'),('moneda',currency),('tarjeta',selected or 'Todas'),('categoria',category or 'Todas'),('comercio',merchant or 'Todos'),('consumos','Filtrados por fecha de compra; valor original sin repetir cuotas'),('extractos','Filtrados por fecha de corte; detalle completo del extracto'),('conciliaciones','Incluye los cruces de pagos o abonos seleccionados, aunque su contraparte sea de otro mes')]),
        ]
        return workbook_response(sheets,'tarjetas_y_conciliaciones.xlsx')
    categories=[dict(row,label=dict(CardMovement.CATEGORIES).get(row['category'],row['category'])) for row in expense.values('category').annotate(total=Sum('amount')).order_by('-total')]
    merchants=list(expense.values('merchant').annotate(total=Sum('amount')).order_by('-total')[:15])
    monthly=list(expense.annotate(month=TruncMonth('date')).values('month').annotate(total=Sum('amount')).order_by('month'))
    bank_monthly={r['month'].strftime('%Y-%m'):abs(r['total']) for r in bank.annotate(month=TruncMonth('date')).values('month').annotate(total=Sum('amount')) if r['month']}
    monthly_map={r['month'].strftime('%Y-%m'):r['total'] for r in monthly}
    labels=sorted(set(monthly_map)|set(bank_monthly))
    bank_list=[]
    for tx in bank.order_by('-date','-pk'):
        tx.remaining=bank_remaining(tx)
        if tx.remaining:bank_list.append(tx)
    payments=[]
    for payment in movements.filter(kind='payment').order_by('-date','-pk'):
        payment.remaining=card_remaining(payment)
        if payment.remaining:payments.append(payment)
    # Manual reconciliation can cross any months and can identify previously unclassified outflows.
    manual_banks=Transaction.objects.filter(user=request.user,amount__lt=0,currency=currency).prefetch_related('card_allocations').order_by('-date','-pk')
    search=request.GET.get('bank_search','').strip()
    if search:manual_banks=manual_banks.filter(description__icontains=search)
    if start:manual_banks=manual_banks.filter(date__gte=start,date__lt=end)
    manual_choices=[]
    for tx in manual_banks[:600]:
        tx.remaining=bank_remaining(tx)
        if tx.remaining:manual_choices.append(tx)
    all_payments=[]
    for payment in CardMovement.objects.filter(card__user=request.user,kind='payment',currency=currency).select_related('card').prefetch_related('allocations').order_by('-date','-pk'):
        payment.remaining=card_remaining(payment)
        if payment.remaining:all_payments.append(payment)
    allocations=CardPaymentAllocation.objects.filter(bank_transaction__user=request.user).select_related('bank_transaction','card_payment__card').order_by('-created_at')
    expense_choices=[]
    for entry in expense.filter(kind__in=['purchase','interest','fee','tax'],amount__gt=0).order_by('-date','-pk')[:600]:
        entry.remaining=expense_remaining(entry)
        if entry.remaining:expense_choices.append(entry)
    expense_banks=[]
    for tx in Transaction.objects.filter(user=request.user,amount__lt=0,currency=currency).select_related('movement_type').prefetch_related('expense_allocations').order_by('-date','-pk'):
        if (tx.movement_type_id and tx.movement_type.name=='pago tarjeta credito') or (search and search.lower() in tx.description.lower()):
            tx.remaining=bank_expense_remaining(tx)
            if tx.remaining:expense_banks.append(tx)
    expense_allocations=CardExpenseAllocation.objects.filter(bank_transaction__user=request.user).select_related('bank_transaction','card_expense__card').order_by('-created_at')[:50]
    context={
        'expense_choices':expense_choices,'expense_banks':expense_banks,'expense_allocations':expense_allocations,
        'cards':cards,'selected_card':selected,'month':month,'currency':currency,'category':category,'merchant':merchant,
        'expense_total':_sum(expense),'purchase_total':_sum(expense.filter(kind='purchase')),
        'refund_total':abs(_sum(expense.filter(kind='refund'))),'cost_total':_sum(expense.filter(kind__in=['interest','fee','tax'])),
        'bank_paid_total':abs(_sum(bank)),'pending_bank_total':sum((r.remaining for r in bank_list),ZERO),
        'categories':categories,'merchants':merchants,'category_choices':CardMovement.CATEGORIES,
        'entries':Paginator(expense.order_by('-date','-pk'),40).get_page(request.GET.get('page')),
        'statements':Paginator(statements,12).get_page(request.GET.get('statement_page')),
        'pending_banks':Paginator(bank_list,30).get_page(request.GET.get('bank_page')),
        'pending_card_payments':Paginator(payments,30).get_page(request.GET.get('payment_page')),
        'pending_bank_count':len(bank_list),'pending_card_count':len(payments),
        'manual_banks':manual_choices,'manual_payments':all_payments,'allocations':allocations[:50],
        'bank_search':search,'chart_data':{
            'categories':{'labels':[r['label'] for r in categories],'values':[float(r['total']) for r in categories]},
            'merchants':{'labels':[r['merchant'] for r in merchants],'values':[float(r['total']) for r in merchants]},
            'monthly':{'labels':labels,'spending':[float(monthly_map.get(k,0)) for k in labels],'payments':[float(bank_monthly.get(k,0)) for k in labels]},
        },'query':request.GET.urlencode(),
    }
    return render(request,'cards/dashboard.html',context)


@login_required
@require_POST
def import_cards(request):
    files=request.FILES.getlist('pdfs')
    if not files:
        messages.error(request,'Selecciona al menos un extracto de tarjeta.');return redirect('cards')
    passwords=[pw.strip() for pw in request.POST.get('passwords','').split(',') if pw.strip()]
    root=Path(settings.MEDIA_ROOT)/'cards'/'incoming';root.mkdir(parents=True,exist_ok=True)
    imported=duplicates=0
    for upload in files:
        path=root/f'{request.user.pk}_{uuid4().hex}.pdf'
        with path.open('wb') as handle:
            for chunk in upload.chunks():handle.write(chunk)
        try:
            parsed=read_card_pdf(path,passwords)
            _,created=store_card_statement(request.user,path,parsed)
            imported+=int(created);duplicates+=int(not created)
            path.unlink(missing_ok=True)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Card import failed')
            messages.error(request,f'{upload.name}: {str(exc)[:250]}')
    matched=auto_reconcile(request.user)
    messages.success(request,f'{imported} extractos importados, {duplicates} ya existentes; {matched} pagos conciliados automáticamente.')
    return redirect('cards')


@login_required
@require_POST
def reconcile_cards(request):
    try:
        if request.POST.get('action')=='auto':
            matched=auto_reconcile(request.user);messages.success(request,f'{matched} pagos conciliados. Los casos ambiguos quedan pendientes.')
        elif request.POST.get('action')=='expenses':
            raw_amount=request.POST.get('amount','').strip()
            amount,count=allocate_expenses(request.user,int(request.POST.get('bank_id','')),request.POST.getlist('expense_ids'),Decimal(raw_amount) if raw_amount else None,request.POST.get('notes',''))
            messages.success(request,f'Se asignaron {amount:,.2f} a {count} gastos. El importe se distribuye por fecha de compra, de más antigua a más reciente.')
        elif request.POST.get('action')=='remove_expense':
            row=get_object_or_404(CardExpenseAllocation,pk=request.POST.get('allocation_id'),bank_transaction__user=request.user)
            row.delete();messages.success(request,'Asignación retirada; el importe está disponible de nuevo.')
        elif request.POST.get('action')=='remove':
            allocation=get_object_or_404(CardPaymentAllocation,pk=request.POST.get('allocation_id'),bank_transaction__user=request.user)
            allocation.delete();messages.success(request,'Conciliación retirada. Ambos saldos están disponibles de nuevo.')
        else:
            allocate_payment(request.user,int(request.POST.get('bank_id','')),int(request.POST.get('payment_id','')),Decimal(request.POST.get('amount','')),notes=request.POST.get('notes',''))
            messages.success(request,'Conciliación guardada.')
    except (ValueError,InvalidOperation,Transaction.DoesNotExist,CardMovement.DoesNotExist) as exc:messages.error(request,str(exc))
    return redirect(reverse('cards')+'#reconciliation')


@login_required
@require_POST
def classify_card_movement(request,pk):
    movement=get_object_or_404(CardMovement,pk=pk,card__user=request.user)
    category=request.POST.get('category','');merchant=request.POST.get('merchant','').strip()
    if category not in dict(CardMovement.CATEGORIES) or not merchant or len(merchant)>255:
        messages.error(request,'Selecciona una categoría y un comercio válido.')
    else:
        movement.category=category;movement.merchant=merchant;movement.manually_classified=True
        movement.save(update_fields=['category','merchant','manually_classified']);messages.success(request,'Clasificación actualizada en todos los extractos de esta compra.')
    next_url=request.POST.get('next','')
    if next_url.startswith('/cards/') and url_has_allowed_host_and_scheme(next_url,allowed_hosts={request.get_host()}):
        return redirect(next_url)
    return redirect('cards')


@login_required
def card_statement(request,pk):
    statement=get_object_or_404(CardStatement.objects.select_related('card'),pk=pk,card__user=request.user)
    return render(request,'cards/statement.html',{'statement':statement,'lines':statement.lines.select_related('movement').order_by('movement__date','pk')})


@login_required
@require_POST
def review_card_statement(request,pk):
    statement=get_object_or_404(CardStatement,pk=pk,card__user=request.user)
    statement.reviewed=True;statement.save(update_fields=['reviewed'])
    messages.success(request,'Extracto marcado como revisado.');return redirect('card_statement',pk=pk)


@login_required
def card_source(request,pk):
    statement=get_object_or_404(CardStatement,pk=pk,card__user=request.user)
    root=(Path(settings.MEDIA_ROOT)/'cards').resolve();path=(root/statement.source_pdf).resolve()
    if path.parent!=root or not path.is_file():raise Http404('No está disponible el extracto original.')
    response=FileResponse(path.open('rb'),content_type='application/pdf');response['Cache-Control']='private, no-store';return response
