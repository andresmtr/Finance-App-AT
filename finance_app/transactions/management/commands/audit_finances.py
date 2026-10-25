"""Auditable, conservative repairs. Never infer signs or delete possible duplicates."""
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from finance_app.core.services.classification import get_classification, normalize, special_classification
from finance_app.transactions.models import MovementType, StagedTransaction, Transaction


class Command(BaseCommand):
    help = "Audit all transactions; --apply repairs only unambiguous purpose classifications, with SQLite backup."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--output-dir", default="data/audits")

    def handle(self, *args, **options):
        root = Path(options["output_dir"])
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report = {"applied": options["apply"], "changes": [], "warnings": [], "totals_before": {}, "totals_after": {}}
        updates = []
        for model in (Transaction, StagedTransaction):
            duplicates = defaultdict(list)
            missing = defaultdict(int)
            totals_before = defaultdict(Decimal)
            totals_after = defaultdict(Decimal)
            for row in model.objects.select_related("movement_type").order_by("pk").iterator():
                name = row.movement_type.name if row.movement_type else ""
                text = normalize(row.description)
                special = special_classification(row.description)
                target_type = special
                if "cdt" in text and ("interes" in text or "rendim" in text):
                    target_type = "intereses"
                target_class = row.classification
                if target_type:
                    target_class = get_classification(target_type, row.amount, row.description)
                changes = {}
                if target_type and target_type != name:
                    changes["movement_type"] = {"before": name, "after": target_type}
                if target_class != row.classification:
                    changes["classification"] = {"before": row.classification, "after": target_class}
                if changes:
                    report["changes"].append({"model": model.__name__, "id": row.pk, "description": row.description, "amount": str(row.amount), "fields": changes})
                    updates.append((model, row.pk, row.classification, row.movement_type_id, target_class, target_type))
                key = f"{row.currency}:{row.classification}"
                totals_before[key] += row.amount or Decimal(0)
                totals_after[f"{row.currency}:{target_class}"] += row.amount or Decimal(0)
                reasons = []
                if row.amount is None or row.date is None:
                    reasons.append("missing_date_or_amount")
                if row.amount and ((row.classification == "gasto" and row.amount > 0) or (row.classification == "ingreso" and row.amount < 0)):
                    reasons.append("classification_sign_conflict")
                if row.amount and row.amount < 0 and text.startswith("abono"):
                    reasons.append("negative_abono_requires_source")
                if reasons:
                    report["warnings"].append({"model": model.__name__, "id": row.pk, "amount": str(row.amount), "description": row.description, "reasons": reasons})
                if row.source_pdf and not (Path(settings.MEDIA_ROOT) / "pdfs" / row.source_pdf).is_file():
                    missing[row.source_pdf] += 1
                if model is Transaction:
                    duplicates[(row.user_id, row.bank, row.account_last4, row.date, row.amount, row.currency, row.description)].append(row.pk)
            report[model.__name__] = {"missing_source_pdfs": dict(missing), "possible_duplicate_groups": [ids for ids in duplicates.values() if len(ids) > 1]}
            report["totals_before"][model.__name__] = {key: str(value) for key, value in totals_before.items()}
            report["totals_after"][model.__name__] = {key: str(value) for key, value in totals_after.items()}
        if options["apply"]:
            database = settings.DATABASES["default"]
            if database["ENGINE"] != "django.db.backends.sqlite3":
                raise CommandError("An explicit database backup is required for a non-SQLite database.")
            backup = root / f"before_{stamp}.sqlite3"
            with sqlite3.connect(str(database["NAME"])) as source, sqlite3.connect(str(backup)) as dest:
                source.backup(dest)
            report["backup"] = str(backup)
            # Persist the exact intended changes before mutating any records.
            (root / f"plan_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            with transaction.atomic():
                for model, pk, old_class, old_type, classification, type_name in updates:
                    fields = {"classification": classification}
                    if type_name:
                        fields["movement_type"] = MovementType.objects.get_or_create(name=type_name, defaults={"is_active": True})[0]
                    changed = model.objects.filter(pk=pk, classification=old_class, movement_type_id=old_type).update(**fields)
                    if changed != 1:
                        raise CommandError("Concurrent modification detected; repairs rolled back.")
        path = root / f"audit_{stamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        self.stdout.write(f"{len(updates)} corrections; {len(report['warnings'])} warnings. Report: {path}")
