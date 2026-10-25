import json
import os
from datetime import datetime
from pathlib import Path
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from finance_app.transactions.services.card_statements import read_card_pdf,store_card_statement
from finance_app.transactions.services.card_reconciliation import auto_reconcile


class Command(BaseCommand):
    help='Import credit-card PDFs into the separate card ledger; preserve cash-flow transactions.'
    def add_arguments(self,parser):
        parser.add_argument('--refresh',action='store_true',help='Re-read existing periods; preserve unchanged movement classifications and matches.')
        parser.add_argument('--manifest',required=True,help='Private JSON file containing local PDF paths or objects with a local key.')
        parser.add_argument('--user-id',required=True,type=int)
        parser.add_argument('--password-env',default='CARD_PDF_PASSWORD')
    def handle(self,*args,**options):
        user=get_user_model().objects.get(pk=options['user_id']);items=json.loads(Path(options['manifest']).read_text())
        password=os.environ.get(options['password_env'],'');report={'imported':[],'duplicates':[],'errors':[]}
        for item in items:
            path=Path(item['local'] if isinstance(item,dict) else item)
            try:
                parsed=read_card_pdf(path,[password] if password else [])
                statement,created=store_card_statement(user,path,parsed,refresh=options['refresh'])
                report['imported' if created else 'duplicates'].append({'file':path.name,'statement_id':statement.pk,'rows':len(parsed['rows']),'notes':parsed['validation_notes']})
            except Exception as exc:report['errors'].append({'file':path.name,'error':str(exc)})
        report['matched']=auto_reconcile(user)
        root=Path('data/audits');root.mkdir(parents=True,exist_ok=True);path=root/f'card_import_{datetime.now():%Y%m%d_%H%M%S}.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        self.stdout.write(f"Imported {len(report['imported'])}; duplicates {len(report['duplicates'])}; errors {len(report['errors'])}; matched {report['matched']}. Report: {path}")
