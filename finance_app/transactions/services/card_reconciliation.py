"""Allocate bank outflows to card statement payments; never to individual purchases."""
import re
from datetime import timedelta
from decimal import Decimal

from django.db import transaction
from django.db.models import Q, Sum

from finance_app.transactions.models import CardMovement, CardPaymentAllocation, MovementType, Transaction
from finance_app.core.services.classification import normalize

ZERO=Decimal('0.00')


def _used(instance,relation):
    if relation in getattr(instance,'_prefetched_objects_cache',{}):
        return sum((row.amount for row in getattr(instance,relation).all()),ZERO)
    return getattr(instance,relation).aggregate(total=Sum('amount'))['total'] or ZERO



def bank_payments(user):
    return Transaction.objects.filter(user=user,amount__lt=0).filter(
        Q(movement_type__name='pago tarjeta credito') |
        Q(description__iregex=r'pago.*(tarj|t\.c\.)') |
        Q(card_allocations__isnull=False)).distinct().prefetch_related('card_allocations','expense_allocations')


def bank_remaining(tx):
    used=_used(tx,'card_allocations')
    return max(abs(tx.amount)-used,ZERO)


def card_remaining(payment):
    used=_used(payment,'allocations')
    return max(abs(payment.amount)-used,ZERO)


def destination_last4(description):
    text=normalize(description)
    match=re.search(r'(?:n\.?\s*|#\s*)?(\d{12,19})\b',text)
    if match:return match[1][-4:]
    match=re.search(r'[*x]{4}[\s*x-]*(\d{4})\b',text)
    return match[1] if match else None


@transaction.atomic
def allocate_payment(user,bank_id,payment_id,amount,method='manual',notes=''):
    tx=Transaction.objects.select_for_update().get(pk=bank_id,user=user)
    payment=CardMovement.objects.select_for_update().get(pk=payment_id,card__user=user,kind='payment')
    amount=Decimal(str(amount))
    if not amount.is_finite() or amount<=0 or amount!=amount.quantize(Decimal('.01')):raise ValueError('El valor a conciliar debe ser positivo y tener como máximo dos decimales.')
    if tx.amount>=0 or payment.amount>=0:raise ValueError('Solo se concilian salidas de cuenta con abonos de la tarjeta.')
    if tx.currency!=payment.currency:raise ValueError('Las monedas del pago y el extracto no coinciden.')
    if amount>bank_remaining(tx) or amount>card_remaining(payment):raise ValueError('El valor supera el saldo pendiente del pago o del abono.')
    allocation,created=CardPaymentAllocation.objects.get_or_create(bank_transaction=tx,card_payment=payment,defaults={'amount':amount,'method':method,'notes':notes[:255]})
    if not created:
        allocation.amount+=amount;allocation.method=method;allocation.save(update_fields=['amount','method'])
    if method=='manual':
        category,_=MovementType.objects.get_or_create(name='pago tarjeta credito',defaults={'is_active':True})
        tx.movement_type=category;tx.save(update_fields=['movement_type'])
    return allocation


def auto_reconcile(user):
    banks=list(bank_payments(user).exclude(date=None).order_by('date','pk'))
    payments=list(CardMovement.objects.filter(card__user=user,kind='payment',amount__lt=0).select_related('card').prefetch_related('allocations').order_by('date','pk'))
    bank_available={tx.pk:bank_remaining(tx) for tx in banks};card_available={p.pk:card_remaining(p) for p in payments}
    candidates={};reverse={}
    for tx in banks:
        available=bank_available[tx.pk]
        if not available:continue
        hint=destination_last4(tx.description)
        options=[p for p in payments if card_available[p.pk]==available and p.currency==tx.currency and abs((p.date-tx.date).days)<=7 and (not hint or p.card.last4==hint)]
        candidates[tx.pk]=options
        for p in options:reverse.setdefault(p.pk,[]).append(tx)
    matched=0
    for tx in banks:
        options=candidates.get(tx.pk,[])
        if len(options)==1 and len(reverse[options[0].pk])==1:
            allocate_payment(user,tx.pk,options[0].pk,bank_available[tx.pk],method='auto',notes='Importe exacto; candidato único en ambos lados; fecha ±7 días.')
            matched+=1
    return matched


def expense_remaining(expense):
    used=_used(expense,'expense_allocations')
    return max(expense.amount-used,ZERO)


def bank_expense_remaining(tx):
    used=_used(tx,'expense_allocations')
    return max(abs(tx.amount)-used,ZERO)


@transaction.atomic
def allocate_expenses(user,bank_id,expense_ids,amount=None,notes=''):
    """Manual purchase attribution is separate from matching the card's payment credit."""
    from finance_app.transactions.models import CardExpenseAllocation
    tx=Transaction.objects.select_for_update().get(pk=bank_id,user=user)
    ids=set(int(pk) for pk in expense_ids)
    if not ids:raise ValueError('Selecciona al menos una compra o gasto.')
    expenses=list(CardMovement.objects.select_for_update().filter(pk__in=ids,card__user=user,kind__in=['purchase','interest','fee','tax'],amount__gt=0).order_by('date','pk'))
    if len(expenses)!=len(ids):raise ValueError('La selección incluye gastos no disponibles o de otra cuenta.')
    if tx.amount>=0 or any(e.currency!=tx.currency for e in expenses):raise ValueError('El pago debe ser una salida y los gastos deben estar en la misma moneda.')
    available=bank_expense_remaining(tx)
    balances={e.pk:expense_remaining(e) for e in expenses}
    total=sum(balances.values(),ZERO)
    amount=min(total,available) if amount is None else Decimal(str(amount))
    if not amount.is_finite() or amount<=0 or amount!=amount.quantize(Decimal('.01')):raise ValueError('Escribe un importe positivo con máximo dos decimales.')
    if amount>available or amount>total:raise ValueError('El importe supera el valor del pago o el saldo de los gastos seleccionados.')
    left=amount;count=0
    for expense in expenses:
        part=min(left,balances[expense.pk])
        if not part:continue
        row,created=CardExpenseAllocation.objects.get_or_create(bank_transaction=tx,card_expense=expense,defaults={'amount':part,'notes':notes[:255]})
        if not created:row.amount+=part;row.save(update_fields=['amount'])
        left-=part;count+=1
        if left==0:break
    category,_=MovementType.objects.get_or_create(name='pago tarjeta credito',defaults={'is_active':True})
    tx.movement_type=category;tx.save(update_fields=['movement_type'])
    return amount,count
