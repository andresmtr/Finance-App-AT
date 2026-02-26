from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

from finance_app.core.services.pdf_ingestion import get_classification
from finance_app.transactions.models import StagedTransaction, Transaction


def _movement_name(instance) -> str:
    return instance.movement_type.name if instance.movement_type else ""


class Command(BaseCommand):
    help = "Recalculate classification for Transaction and StagedTransaction."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only show how many rows would change.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]

        tx_changed, tx_scanned = self._reclassify_queryset(
            Transaction.objects.select_related("movement_type").all(),
            dry_run=dry_run,
            model_name="Transaction",
        )
        staged_changed, staged_scanned = self._reclassify_queryset(
            StagedTransaction.objects.select_related("movement_type").all(),
            dry_run=dry_run,
            model_name="StagedTransaction",
        )

        self.stdout.write(
            self.style.SUCCESS(
                "Done. "
                f"Transaction: {tx_changed}/{tx_scanned} changed. "
                f"StagedTransaction: {staged_changed}/{staged_scanned} changed."
            )
        )

    def _reclassify_queryset(self, queryset, *, dry_run: bool, model_name: str) -> tuple[int, int]:
        to_update = []
        scanned = 0

        for row in queryset.iterator(chunk_size=1000):
            scanned += 1
            new_classification = get_classification(_movement_name(row), row.amount)
            if row.classification != new_classification:
                row.classification = new_classification
                to_update.append(row)

        changed = len(to_update)
        if changed == 0:
            self.stdout.write(f"{model_name}: no changes needed.")
            return 0, scanned

        if dry_run:
            self.stdout.write(f"{model_name}: {changed} rows would be updated.")
            return changed, scanned

        with transaction.atomic():
            queryset.model.objects.bulk_update(to_update, ["classification"], batch_size=1000)
        self.stdout.write(f"{model_name}: {changed} rows updated.")
        return changed, scanned
