"""Classification uses the description for purpose and the signed amount for direction."""
import re
import unicodedata
from decimal import Decimal


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
