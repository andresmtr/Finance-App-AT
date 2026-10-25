"""Apply only the explicitly requested spending classification rules."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand,CommandError
from django.db import transaction

from finance_app.transactions.models import CardMovement
from finance_app.transactions.services.card_statements import household_category


class Command(BaseCommand):
    help='Review or apply requested pharmacy, OXXO, utility, household and education categories; keep amounts and allocations unchanged.'

    def add_arguments(self,parser):
        parser.add_argument('--apply',action='store_true')
        parser.add_argument('--output-dir',default='data/audits')

    def handle(self,*args,**options):
        changes=[]
        for movement in CardMovement.objects.filter(kind__in=['purchase','refund']).order_by('pk').iterator():
            target=household_category(movement.description,movement.kind) or household_category(movement.merchant,movement.kind)
            if target and target!=movement.category:
                changes.append({'id':movement.pk,'before':movement.category,'after':target,'amount':str(movement.amount),'merchant':movement.merchant,'description':movement.description})
        root=Path(options['output_dir']);root.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');path=root/f'card_categories_{stamp}.json'
        report={'applied':False,'changes':changes};path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        if options['apply']:
            database=settings.DATABASES['default']
            if database['ENGINE']!='django.db.backends.sqlite3':raise CommandError('Se requiere respaldo previo para bases distintas de SQLite.')
            backup=root/f'before_card_categories_{stamp}.sqlite3'
            with sqlite3.connect(str(database['NAME'])) as source,sqlite3.connect(str(backup)) as target:source.backup(target)
            with transaction.atomic():
                for change in changes:
                    count=CardMovement.objects.filter(pk=change['id'],category=change['before'],amount=change['amount'],description=change['description'],merchant=change['merchant']).update(category=change['after'])
                    if count!=1:raise CommandError('Un movimiento cambió durante la revisión; se revirtieron las categorías.')
            report.update(applied=True,backup=str(backup));path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        self.stdout.write(f'{len(changes)} category changes; applied={report["applied"]}. Report: {path}')
