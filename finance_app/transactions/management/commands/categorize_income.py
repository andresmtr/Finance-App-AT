import json
import sqlite3
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finance_app.core.services.classification import EARNED_INCOME_TYPE, refine_movement_type
from finance_app.transactions.models import MovementType, StagedTransaction, Transaction


class Command(BaseCommand):
    help = "Identify salary/fee income and correct negative abono categories, without changing amounts. Defaults to dry run."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--income-keyword", action="append", default=[], help="Confirmed payer phrase to remember for future PDF imports.")
        parser.add_argument("--output-dir", default="data/audits")

    def handle(self, *args, **options):
        income_type = MovementType.objects.filter(name=EARNED_INCOME_TYPE).first()
        keywords = list(dict.fromkeys((income_type.keywords if income_type else []) + options["income_keyword"]))
        changes = []
        totals = {}
        for model in (Transaction, StagedTransaction):
            qs = model.objects.select_related("movement_type")
            if model is StagedTransaction:
                qs = qs.filter(status__in=["pending", "approved"])
            count = 0
            for row in qs.order_by("pk").iterator():
                old_type = row.movement_type.name if row.movement_type else ""
                new_type = refine_movement_type(row.description, row.amount, old_type, keywords, row.date)
                if old_type != new_type:
                    changes.append({"model": model.__name__, "id": row.pk, "before": old_type,
                                    "before_id": row.movement_type_id, "after": new_type,
                                    "amount": str(row.amount), "classification": row.classification,
                                    "description": row.description})
                    count += 1
            totals[model.__name__] = count
        root = Path(options["output_dir"])
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report = {"applied": False, "income_keywords": keywords, "counts": totals, "changes": changes}
        plan = root / f"income_categories_{stamp}.json"
        plan.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if options["apply"]:
            database = settings.DATABASES["default"]
            if database["ENGINE"] != "django.db.backends.sqlite3":
                raise CommandError("A database backup is required before applying changes outside SQLite.")
            backup = root / f"before_income_categories_{stamp}.sqlite3"
            with sqlite3.connect(str(database["NAME"])) as source, sqlite3.connect(str(backup)) as target:
                source.backup(target)
            report["backup"] = str(backup)
            with transaction.atomic():
                category, _ = MovementType.objects.get_or_create(name=EARNED_INCOME_TYPE)
                category.keywords = keywords
                category.is_active = True
                category.save(update_fields=["keywords", "is_active"])
                models = {"Transaction": Transaction, "StagedTransaction": StagedTransaction}
                for change in changes:
                    target, _ = MovementType.objects.get_or_create(name=change["after"], defaults={"is_active": True})
                    updated = models[change["model"]].objects.filter(
                        pk=change["id"], movement_type_id=change["before_id"],
                        amount=change["amount"], description=change["description"],
                    ).update(movement_type=target)
                    if updated != 1:
                        raise CommandError("Concurrent modification detected; changes rolled back.")
            report["applied"] = True
            plan.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        self.stdout.write(f"{totals}. Applied: {report['applied']}. Report: {plan}")
