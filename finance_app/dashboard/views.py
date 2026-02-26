import pandas as pd
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from finance_app.transactions.models import Transaction


def _safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.fillna(0).sum())


def _apply_filters(
    frame: pd.DataFrame,
    account_last4: str,
    classification: str,
    movement_type: str,
    bank: str,
    reference: str,
) -> pd.DataFrame:
    filtered = frame
    if bank:
        filtered = filtered[filtered["bank"] == bank]
    if account_last4:
        filtered = filtered[filtered["account_last4"] == account_last4]
    if classification:
        filtered = filtered[filtered["classification"] == classification]
    if movement_type:
        filtered = filtered[filtered["movement_type"] == movement_type]
    if reference:
        filtered = filtered[filtered["reference"].str.contains(reference, case=False, na=False)]
    return filtered


def _format_period_labels(index: pd.DatetimeIndex, granularity: str) -> list[str]:
    if granularity == "monthly":
        return index.strftime("%b %Y").tolist()
    return [value.date().isoformat() for value in index]


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    base_qs = (
        Transaction.objects.filter(user=request.user)
        .select_related("movement_type")
        .values(
            "bank",
            "account_last4",
            "date",
            "time",
            "amount",
            "currency",
            "movement_type__name",
            "classification",
            "reference",
            "merchant",
            "location",
            "channel",
            "description",
            "external_id",
        )
    )
    frame_all = pd.DataFrame.from_records(base_qs)
    if frame_all.empty:
        frame_all = pd.DataFrame(
            columns=[
                "bank",
                "account_last4",
                "date",
                "time",
                "amount",
                "currency",
                "movement_type",
                "classification",
                "reference",
                "merchant",
                "location",
                "channel",
                "description",
                "id",
            ]
        )
    else:
        frame_all = frame_all.rename(columns={"movement_type__name": "movement_type"})
        frame_all["id"] = frame_all["external_id"]
        frame_all = frame_all.drop(columns=["external_id"])
        frame_all = frame_all.dropna(subset=["amount"])
        frame_all["amount"] = frame_all["amount"].astype(float)

    bank = request.GET.get("bank", "")
    account_last4 = request.GET.get("account_last4", "")
    classification = request.GET.get("classification", "")
    movement_type = request.GET.get("movement_type", "")
    reference = request.GET.get("reference", "")
    year_filter = request.GET.get("year", "")
    granularity = "monthly"
    freq = "ME"
    classification_choices = list(Transaction._meta.get_field("classification").choices)
    classification_keys = [choice[0] for choice in classification_choices]
    classification_labels = {choice[0]: choice[1] for choice in classification_choices}

    # Extract available years for the dropdown before filtering
    available_years = []
    if not frame_all.empty:
        frame_all["date_dt"] = pd.to_datetime(frame_all["date"], errors="coerce")
        frame_all = frame_all.dropna(subset=["date_dt"])
        available_years = sorted(frame_all["date_dt"].dt.year.unique().astype(int).tolist(), reverse=True)

    if year_filter and year_filter.isdigit():
        frame_all = frame_all[frame_all["date_dt"].dt.year == int(year_filter)]

    frame = _apply_filters(frame_all, account_last4, classification, movement_type, bank, reference)
    frame = frame.copy()

    if not frame.empty:
        income_mask = frame["classification"] == "ingreso"
        expense_mask = frame["classification"] == "gasto"
        frame["income_flag"] = income_mask
        frame["expense_flag"] = expense_mask

    income = frame[frame["income_flag"]] if not frame.empty else frame
    expense = frame[frame["expense_flag"]] if not frame.empty else frame

    total_income = _safe_sum(income["amount"]) if not income.empty else 0.0
    total_expense = _safe_sum(expense["amount"].abs()) if not expense.empty else 0.0
    net = total_income - total_expense

    total_cdt_in = _safe_sum(frame[(frame["classification"] == "cdt") & (frame["amount"] > 0)]["amount"]) if not frame.empty else 0.0
    total_cdt_out = _safe_sum(frame[(frame["classification"] == "cdt") & (frame["amount"] < 0)]["amount"].abs()) if not frame.empty else 0.0
    
    total_bolsillo_in = _safe_sum(frame[(frame["classification"] == "bolsillo") & (frame["amount"] > 0)]["amount"]) if not frame.empty else 0.0
    total_bolsillo_out = _safe_sum(frame[(frame["classification"] == "bolsillo") & (frame["amount"] < 0)]["amount"].abs()) if not frame.empty else 0.0

    tx_count = int(len(frame.index)) if not frame.empty else 0

    # Helper: Use merchant if available, else first 30 chars of description
    if not frame.empty:
        frame["display_entity"] = frame.apply(
            lambda x: x["merchant"] if x.get("merchant") else (str(x["description"])[:35] + "..." if len(str(x["description"])) > 35 else str(x["description"])),
            axis=1
        )
        income = frame[frame["income_flag"]]
        expense = frame[frame["expense_flag"]]

    merchant_spend = (
        expense.assign(amount_abs=expense["amount"].abs())
        .groupby("display_entity")["amount_abs"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        if not expense.empty
        else pd.Series()
    )
    merchant_labels = [m if m else "Desconocido" for m in merchant_spend.index.fillna("N/A").tolist()]
    merchant_values = merchant_spend.fillna(0).astype(float).tolist()
    
    # Top Income Sources
    top_income = (
        income.groupby("display_entity")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        if not income.empty
        else pd.Series()
    )
    top_income_labels = [m if m else "Desconocido" for m in top_income.index.fillna("N/A").tolist()]
    top_income_values = top_income.fillna(0).astype(float).tolist()

    if not frame.empty:
        movement_group = (
            frame.assign(amount_abs=frame["amount"].abs())
            .groupby("classification")["amount_abs"]
            .sum()
            .reindex(classification_keys, fill_value=0)
        )
    else:
        movement_group = pd.Series(0.0, index=classification_keys, dtype=float)
    movement_labels = [classification_labels.get(key, key.title()) for key in movement_group.index.tolist()]
    movement_values = movement_group.fillna(0).astype(float).tolist()

    type_group = (
        frame.assign(amount_abs=frame["amount"].abs())
        .groupby("movement_type")["amount_abs"]
        .sum()
        .sort_values(ascending=False)
        if not frame.empty
        else pd.Series()
    )
    type_labels = type_group.index.fillna("N/A").astype(str).tolist()
    type_values = type_group.fillna(0).astype(float).tolist()

    # Bank Group: Split into Income and Expense per bank
    if not frame.empty:
        bank_inc = income.groupby("bank")["amount"].sum()
        bank_exp = expense.groupby("bank")["amount"].sum().abs()
        all_banks = list(set(bank_inc.index.tolist() + bank_exp.index.tolist()))
        bank_inc = bank_inc.reindex(all_banks, fill_value=0)
        bank_exp = bank_exp.reindex(all_banks, fill_value=0)
        bank_labels = [str(b) if b else "Desconocido" for b in all_banks]
        bank_inc_vals = bank_inc.fillna(0).astype(float).tolist()
        bank_exp_vals = bank_exp.fillna(0).astype(float).tolist()
    else:
        bank_labels = []
        bank_inc_vals = []
        bank_exp_vals = []

    if not frame.empty:
        indexed = frame.set_index("date_dt").sort_index()
        net_series = indexed["amount"].resample(freq).sum()
        income_series = indexed[indexed["income_flag"]]["amount"].resample(freq).sum()
        expense_series = indexed[indexed["expense_flag"]]["amount"].resample(freq).sum().abs()
        income_series = income_series.reindex(net_series.index, fill_value=0)
        expense_series = expense_series.reindex(net_series.index, fill_value=0)
        # Classification Series
        # Ingreso
        class_ingreso_series = indexed[indexed["classification"] == "ingreso"]["amount"].resample(freq).sum()
        class_ingreso_series = class_ingreso_series.reindex(net_series.index, fill_value=0)
        
        # Gasto
        class_gasto_series = indexed[indexed["classification"] == "gasto"]["amount"].resample(freq).sum().abs()
        class_gasto_series = class_gasto_series.reindex(net_series.index, fill_value=0)
        
        # CDT
        class_cdt_series = indexed[indexed["classification"] == "cdt"]["amount"].resample(freq).sum()
        class_cdt_series = class_cdt_series.reindex(net_series.index, fill_value=0)

        # Bolsillo
        class_bolsillo_series = indexed[indexed["classification"] == "bolsillo"]["amount"].resample(freq).sum()
        class_bolsillo_series = class_bolsillo_series.reindex(net_series.index, fill_value=0)

        ts_labels = _format_period_labels(net_series.index, granularity)
        
        # New: Monthly aggregated income vs expense
        monthly_inc = indexed[indexed["income_flag"]]["amount"].resample("ME").sum()
        monthly_exp = indexed[indexed["expense_flag"]]["amount"].resample("ME").sum().abs()
        monthly_labels_list = net_series.index
        monthly_inc = monthly_inc.reindex(monthly_labels_list, fill_value=0)
        monthly_exp = monthly_exp.reindex(monthly_labels_list, fill_value=0)
        inc_exp_labels = monthly_labels_list.strftime("%b %Y").tolist()
        inc_exp_inc_vals = monthly_inc.fillna(0).astype(float).tolist()
        inc_exp_exp_vals = monthly_exp.fillna(0).astype(float).tolist()
    else:
        net_series = pd.Series()
        income_series = pd.Series()
        expense_series = pd.Series()
        class_ingreso_series = pd.Series()
        class_gasto_series = pd.Series()
        class_cdt_series = pd.Series()
        class_bolsillo_series = pd.Series()
        ts_labels = []
        inc_exp_labels = []
        inc_exp_inc_vals = []
        inc_exp_exp_vals = []

    monthly_net = (
        frame.groupby(pd.Grouper(key="date_dt", freq="ME"))["amount"].sum().sort_index()
        if not frame.empty
        else pd.Series()
    )
    if not monthly_net.empty:
        monthly_net = monthly_net.dropna()
        monthly_avg = monthly_net
        # Cumulative flow shown by month-year
        cum_net = monthly_net.cumsum()
        cum_labels = cum_net.index.strftime("%b %Y").tolist()
        cum_values = cum_net.fillna(0).astype(float).tolist()
    else:
        monthly_avg = pd.Series()
        cum_labels = []
        cum_values = []
        
    avg_monthly_net = float(monthly_avg.mean()) if not monthly_avg.empty else 0.0

    chart_data = {
        "time_series": {
            "labels": ts_labels,
            "net": net_series.fillna(0).astype(float).tolist() if not net_series.empty else [],
            "income": income_series.fillna(0).astype(float).tolist() if not income_series.empty else [],
            "expense": expense_series.fillna(0).astype(float).tolist() if not expense_series.empty else [],
            "class_ingreso": class_ingreso_series.fillna(0).astype(float).tolist() if not class_ingreso_series.empty else [],
            "class_gasto": class_gasto_series.fillna(0).astype(float).tolist() if not class_gasto_series.empty else [],
            "class_cdt": class_cdt_series.fillna(0).astype(float).tolist() if not class_cdt_series.empty else [],
            "class_bolsillo": class_bolsillo_series.fillna(0).astype(float).tolist() if not class_bolsillo_series.empty else [],
        },
        "movement": {"labels": movement_labels, "values": movement_values},
        "types": {"labels": type_labels, "values": type_values},
        "merchants": {"labels": merchant_labels, "values": merchant_values},
        "banks": {"labels": bank_labels, "income": bank_inc_vals, "expense": bank_exp_vals},
        "top_income": {"labels": top_income_labels, "values": top_income_values},
        "monthly_inc_exp": {
            "labels": inc_exp_labels,
            "income": inc_exp_inc_vals,
            "expense": inc_exp_exp_vals,
        },
        "cumulative": {"labels": cum_labels, "values": cum_values},
    }

    context = {
        "total_income": total_income,
        "total_expense": total_expense,
        "net": net,
        "tx_count": tx_count,
        "avg_monthly_net": avg_monthly_net,
        "total_cdt_in": total_cdt_in,
        "total_cdt_out": total_cdt_out,
        "total_bolsillo_in": total_bolsillo_in,
        "total_bolsillo_out": total_bolsillo_out,
        "chart_data_json": chart_data,
        "filters": request.GET,
        "granularity": granularity,
        "available_years": available_years,
        "selected_year": year_filter,
        "banks": sorted(frame_all["bank"].dropna().unique().tolist()) if not frame_all.empty else [],
        "accounts": sorted(
            frame_all["account_last4"]
            .dropna()
            .loc[frame_all["account_last4"] != "6176"]
            .unique()
            .tolist()
        )
        if not frame_all.empty
        else [],
        "movement_types": sorted(frame_all["movement_type"].dropna().astype(str).unique().tolist()) if not frame_all.empty else [],
        "classifications": classification_keys,
    }
    return render(request, "dashboard/dashboard.html", context)
