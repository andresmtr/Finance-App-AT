"""Classification uses the description for purpose and the signed amount for direction."""
import re
import os
import unicodedata
from decimal import Decimal
from datetime import date

EARNED_INCOME_TYPE = "pago de salarios u honorarios"


def is_education_payment(description):
    text=normalize(description)
    return bool(re.search(r'\b(?:unad|preply|udemy)\b|\bgimnasio moderno castil(?:la|l)?\b',text))


def is_housing_loan(description):
    text=normalize(description)
    if re.search(r'\bcredito (?:de |para )?vivienda\b|\bcredito hipotecario\b|\bprestamo hipotecario\b|\bcuota hipotecaria\b|\bleasing habitacional\b|\bpago.*hipoteca\b',text):
        return True
    configured=[value.strip() for value in os.getenv('HOUSING_CREDIT_LAST4','').split(',') if len(value.strip())==4 and value.strip().isdigit()]
    return bool(re.search(r'\bpago cuota cartera\b',text) and any(re.search(r'\b'+re.escape(last4)+r'\b',text) for last4 in configured))


def movement_override(description, amount=None, income_keywords=(), transaction_date=None):
    """Identify earned income only on incoming movements, never by bank name alone."""
    text = normalize(description)
    if special_classification(description):
        return None
    try:
        value = Decimal(str(amount))
        if not value.is_finite():
            return None
    except (ValueError, ArithmeticError):
        return None
    if value < 0:
        if is_education_payment(description) and not re.search(r'pago.*tarj',text):
            return "educación"
        if is_housing_loan(description):
            return "credito de vivienda"
        # 'Recaudo' on a direct debit is a payment, despite the old generic abono rule.
        if re.search(r"\b(cargo|debito|descuento|dcto)\b", text) and "recaudo" in text:
            return "pago"
        if text.startswith("abono"):
            return "pago"
        return None
    if value == 0 or re.search(r"\b(reembolso|devolucion|reversion|anulacion|intereses?|rendimientos?)\b", text):
        return None
    # Payer identities and the OIM cutoff were confirmed by the account owner.
    if "engineer systems" in text or re.search(r"(?<!\d)0*9009495364(?!\d)", text):
        return EARNED_INCOME_TYPE
    try:
        posted_date = date.fromisoformat(str(transaction_date))
    except (ValueError, TypeError):
        posted_date = None
    if posted_date and posted_date <= date(2021, 12, 31) and re.search(r"\b(citibank|ciban)\b", text):
        return EARNED_INCOME_TYPE
    explicit = re.search(
        r"\b(nomina|honorarios?|contratista|salarios?|sueldos?)\b"
        r"|\bliq(?:uidacion)?\.?\s+contrato\s+de\s+trabajo\b"
        r"|\bpago\s+(?:de\s+)?prima\b", text,
    )
    confirmed_payer = any(normalize(keyword) in text for keyword in income_keywords if normalize(keyword))
    if explicit or confirmed_payer:
        return EARNED_INCOME_TYPE
    return None


def refine_movement_type(description, amount, current_type, income_keywords=(), transaction_date=None):
    override = movement_override(description, amount, income_keywords, transaction_date)
    if override:
        return override
    try:
        if Decimal(str(amount)) < 0 and current_type in {"abono", EARNED_INCOME_TYPE, "pago de salarios", "honorarios OIM"}:
            return "pago"
    except (ValueError, ArithmeticError):
        pass
    return current_type


def normalize(text):
    return " ".join(unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower().split())


def special_classification(description):
    text = normalize(description)
    # These lines move interest already credited by the bank between own accounts.
    if "bolsillo" in text or "rendimientos netos desde cuenta" in text:
        return "bolsillo"
    if re.search(r"\bcdt\b|certificado de deposito", text):
        if "interes" not in text and "rendim" not in text:
            return "cdt"
    return None


def get_classification(movement_name, amount=None, description=""):
    special = special_classification(description)
    if special:
        return special
    text = normalize(description)
    # Interest on an investment is income/expense, separate from its principal.
    is_interest = "interes" in text or "rendim" in text
    if not is_interest:
        special = special_classification(movement_name)
        if special:
            return special
    try:
        value = Decimal(str(amount))
        if value.is_finite() and value > 0:
            return "ingreso"
    except (ValueError, ArithmeticError):
        pass
    return "gasto"
