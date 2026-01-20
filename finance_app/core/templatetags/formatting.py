from django import template

register = template.Library()


def _format_spanish(value: float, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


@register.filter
def money_es(value: float | int | None) -> str:
    if value is None:
        return "0,00"
    try:
        return _format_spanish(float(value), 2)
    except (TypeError, ValueError):
        return "0,00"


@register.filter
def int_es(value: int | float | None) -> str:
    if value is None:
        return "0"
    try:
        formatted = f"{int(value):,d}"
        return formatted.replace(",", ".")
    except (TypeError, ValueError):
        return "0"
