"""Read native statement text without guessing amount signs or using an LLM."""
import re
from datetime import date
from decimal import Decimal

from .classification import normalize

MONEY = r"-?\d[\d,]*\.\d{2}"


def money(value):
    return Decimal(value.replace(",", ""))


def parse_bogota(pages, year):
    text = "\n".join(pages)
    account = re.search(r"Cuenta\s+N[uú]mero:\s*(\d+)", text, re.I)
    if not account or "CodTrans" not in text:
        return None
    rows = []
    for page_number, page in enumerate(pages, 1):
        active = False
        for line in page.splitlines():
            if "CodTrans" in line and "Saldo" in line:
                active = True
                continue
            if "FIN MOVIMIENTOS" in line:
                active = False
            if not active:
                continue
            match = re.match(r"\s*(\d{2})/(\d{2})\s+(\w+)\s+(.+)", line)
            if match:
                values = list(re.finditer(MONEY, line))
                if len(values) < 2:
                    raise ValueError("Transaction line without amount and balance")
                amount, balance = values[-2:]
                parts = re.split(r"\s{2,}", line[match.start(4):amount.start()].strip())
                rows.append({"date": date(year, int(match[2]), int(match[1])).isoformat(),
                             "amount": str(money(amount[0])), "balance": str(money(balance[0])),
                             "description": parts[0], "reference": parts[-1] if len(parts) > 3 else "",
                             "bank": "Banco de Bogotá", "account_last4": account[1][-4:],
                             "currency": "COP", "source_page": page_number})
            elif rows and line.strip() and not re.search(r"Fecha|Documento|pendiente|^\s*[-\d]+\s*$", line):
                # Wrapped descriptions are indented before the city column.
                if len(line) - len(line.lstrip()) < 80:
                    rows[-1]["description"] += " " + line.strip()
    initial = re.search(r"Saldo Inicial:\s*(" + MONEY + ")", text)
    final = re.search(r"Saldo Final:\s*(" + MONEY + ")", text)
    if not initial or not final:
        raise ValueError("Missing statement balances")
    previous = money(initial[1])
    for row in rows:
        if previous + Decimal(row["amount"]) != Decimal(row["balance"]):
            raise ValueError("Running balance does not reconcile")
        previous = Decimal(row["balance"])
    if previous != money(final[1]):
        raise ValueError("Final balance does not reconcile")
    return rows


def parse_davivienda(pages, year):
    text = "\n".join(pages)
    if "CUENTA DE AHORROS" not in text or "Clase de Movimiento" not in text:
        return None
    rows = []
    account = ""
    pocket = False
    for page_number, page in enumerate(pages, 1):
        for line in page.splitlines():
            if "EXTRACTO BOLSILLO" in line:
                pocket = True
            elif "EXTRACTO CUENTA DE AHORROS" in line:
                pocket = False
            match_account = re.search(r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b", line)
            if match_account and "CUENTA DE AHORROS" in page:
                account = match_account[3]
            match = re.match(r"\s*(\d{1,2})\s+(\d{1,2})\s+\$\s*(" + MONEY + r")\s*([+-])\s+(\d+)\s+(.+)", line)
            if match:
                if not account:
                    raise ValueError("Missing account identifier")
                parts = re.split(r"\s{2,}", match[6].strip())
                if pocket and "bolsillo" not in normalize(parts[0]):
                    parts[0] += " (Bolsillo)"
                rows.append({"date": date(year, int(match[2]), int(match[1])).isoformat(),
                             "amount": str(money(match[3]) * (1 if match[4] == "+" else -1)),
                             "description": parts[0], "reference": match[5],
                             "bank": "Davivienda", "account_last4": account,
                             "currency": "COP", "source_page": page_number})
    # Portfolio summaries include the account and its pockets. Reconcile all posted rows.
    initial = re.search(r"Saldo Anterior\s+\$\s*(" + MONEY + ")", text)
    final = re.search(r"Nuevo Saldo\s+\$\s*(" + MONEY + ")", text)
    if not initial or not final or not rows:
        raise ValueError("Missing statement balances or movements")
    primary = rows[0]["account_last4"]
    total = sum((Decimal(row["amount"]) for row in rows if row["account_last4"] == primary), Decimal(0))
    if money(initial[1]) + total != money(final[1]):
        raise ValueError("Final balance does not reconcile")
    return rows


def parse_statement_text(pages, year):
    for parser in (parse_bogota, parse_davivienda):
        rows = parser(pages, year)
        if rows is not None:
            return rows
    raise ValueError("Unsupported statement layout")
