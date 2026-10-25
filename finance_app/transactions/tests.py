import json
from datetime import date
from decimal import Decimal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, SimpleTestCase

from finance_app.core.services.classification import get_classification
from finance_app.core.services.pdf_ingestion import approve_staged_transaction, stage_pdf_files
from finance_app.transactions.models import ImportBatch, MovementType, StagedTransaction, Transaction


class ClassificationTests(SimpleTestCase):
    def test_confirmed_employer_and_oim_cutoff(self):
        from finance_app.core.services.classification import EARNED_INCOME_TYPE, refine_movement_type
        cases = [
            ("Abono Engineer Systems Corp Sas", 100, "2026-01-01", EARNED_INCOME_TYPE),
            ("Transferencia BOGOTA 9009495364", 100, "2019-01-01", EARNED_INCOME_TYPE),
            ("Abono Bco CITIBANK", 100, "2021-12-31", EARNED_INCOME_TYPE),
            ("Abono Cta CIBAN 000000801055528", 100, date(2021, 5, 6), EARNED_INCOME_TYPE),
            ("Abono Bco CITIBANK", 100, "2022-01-01", "transferencia"),
            ("Transferencia CITIBANK", 100, None, "transferencia"),
            ("Transferencia CITIBANK", -100, "2021-12-31", "transferencia"),
            ("Transferencia Engineer Systems Corp Sas", -100, "2026-01-01", "transferencia"),
            ("Reembolso Engineer Systems Corp Sas", 100, "2026-01-01", "transferencia"),
        ]
        for text, amount, posted, expected in cases:
            with self.subTest(text=text, date=posted):
                self.assertEqual(refine_movement_type(text, amount, "transferencia", transaction_date=posted), expected)

    def test_salary_and_fees_require_incoming_work_payment(self):
        from finance_app.core.services.classification import EARNED_INCOME_TYPE, refine_movement_type
        cases = [
            ("Transferencia OCCIDENTE PAGO NOMINA ABRIL", 100, "transferencia", EARNED_INCOME_TYPE),
            ("Transferencia OCCIDENTE Contratista", 100, "transferencia", EARNED_INCOME_TYPE),
            ("Abono honorarios", 100, "abono", EARNED_INCOME_TYPE),
            ("LIQ CONTRATO DE TRABAJO", 100, "transferencia", EARNED_INCOME_TYPE),
            ("PAGO PRIMA DICIEMBRE", 100, "transferencia", EARNED_INCOME_TYPE),
            ("Cargo por Recaudo con Debito Automatico", -20000, "abono", "pago"),
            ("Pago nomina", -100, "transferencia", "transferencia"),
            ("Abono Bco CITIBANK", 100, "abono", "abono"),
            ("Transferencia de un familiar", 100, "transferencia", "transferencia"),
        ]
        for description, amount, current, expected in cases:
            with self.subTest(description=description):
                self.assertEqual(refine_movement_type(description, amount, current), expected)

    def test_confirmed_payers_do_not_override_refunds_interest_or_pockets(self):
        from finance_app.core.services.classification import EARNED_INCOME_TYPE, refine_movement_type
        self.assertEqual(refine_movement_type("Abono CITIBANK", 100, "abono", ["citibank"]), EARNED_INCOME_TYPE)
        for text, amount, current in [
            ("Reembolso examen CITIBANK", 100, "abono"),
            ("Intereses CITIBANK", 100, "intereses"),
            ("Abono de Bolsillo CITIBANK", 100, "bolsillo"),
            ("Pago nomina CITIBANK", -100, "transferencia"),
        ]:
            self.assertEqual(refine_movement_type(text, amount, current, ["citibank"]), current)

    def test_pocket_section_supplies_missing_description_context(self):
        from finance_app.core.services.statement_text import parse_statement_text
        text = """CUENTA DE AHORROS
4744 0006 9974
Saldo Anterior $100.00
Nuevo Saldo $100.00
EXTRACTO CUENTA DE AHORROS
Fecha Valor Doc. Clase de Movimiento
01 01 $50.00- 1234 Debito Automatico Al Bolsillo
EXTRACTO BOLSILLO
01 01 $50.00+ 1234 Abono Automatico Desde Cuenta
"""
        rows = parse_statement_text([text], 2026)
        self.assertEqual(get_classification("abono", rows[-1]["amount"], rows[-1]["description"]), "bolsillo")

    def test_native_statement_requires_balances_to_reconcile(self):
        from finance_app.core.services.statement_text import parse_statement_text
        text = """Cuenta Número: 000109306
Saldo Inicial: 100.00
Saldo Final: 125.00
Fecha CodTrans Descripción Saldo
01/01 0060 Abono transferencia  Bogota  Principal  000000  25.00  125.00
FIN MOVIMIENTOS
"""
        rows = parse_statement_text([text], 2026)
        self.assertEqual(rows[0]["amount"], "25.00")
        self.assertEqual(rows[0]["account_last4"], "9306")
        with self.assertRaises(ValueError):
            parse_statement_text([text.replace("25.00  125.00", "35.00  125.00")], 2026)

    def test_internal_transfers_and_investment_interest(self):
        cases = [
            ("transferencia", -3000000, "Transferencia De Cuenta A Bolsillo", "bolsillo"),
            ("intereses", 127, "Abono Rendimientos Netos Desde Cuenta", "bolsillo"),
            ("cdt", 102489, "Abono de intereses por la cancelacion del CDT desmaterializado", "ingreso"),
            ("cdt", 2000000, "Abono por cancelacion CDT desmaterializado capital", "cdt"),
            ("transferencia", -705426, "Dcto.transferencia otra entidad Abono proceso", "gasto"),
            ("abono", 100000, "Abono transferencia", "ingreso"),
        ]
        for movement, amount, description, expected in cases:
            with self.subTest(description=description):
                self.assertEqual(get_classification(movement, amount, description), expected)


class FinanceTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="audit-test")
        self.client.force_login(self.user)

    def test_approval_cannot_duplicate_or_approve_missing_amount(self):
        batch = ImportBatch.objects.create(user=self.user)
        staged = StagedTransaction.objects.create(user=self.user, batch=batch, date=date(2026, 1, 1), amount=100)
        approve_staged_transaction(staged)
        with self.assertRaises(ValueError):
            approve_staged_transaction(staged)
        self.assertEqual(Transaction.objects.count(), 1)

    def test_source_document_is_private_and_scoped_to_owner(self):
        tx = Transaction.objects.create(user=self.user, amount=1, source_pdf="private.pdf")
        stranger = get_user_model().objects.create_user(username="stranger")
        self.client.force_login(stranger)
        self.assertEqual(self.client.get(f"/transactions/{tx.pk}/source/").status_code, 404)
        self.client.logout()
        self.assertEqual(self.client.get(f"/transactions/{tx.pk}/source/").status_code, 302)

    def test_dashboard_operating_net_matches_chart(self):
        for amount, classification in [(1000, "ingreso"), (-100, "gasto"), (-5000, "bolsillo"), (-2000, "cdt")]:
            Transaction.objects.create(user=self.user, amount=amount, classification=classification, date=date(2026, 1, 1))
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["net"], 900)
        self.assertEqual(response.context["chart_data_json"]["time_series"]["net"], [900])
        self.assertEqual(response.context["chart_data_json"]["cumulative"]["values"], [900])

    def test_audit_dry_run_preserves_ambiguous_signs(self):
        movement = MovementType.objects.get_or_create(name="transferencia")[0]
        tx = Transaction.objects.create(user=self.user, amount=-3000, description="Transferencia De Cuenta A Bolsillo", movement_type=movement, classification="gasto")
        Transaction.objects.create(user=self.user, amount=-17100, description="Abono Transferencia por canal electronico", classification="gasto")
        with TemporaryDirectory() as root:
            call_command("audit_finances", output_dir=root)
            from pathlib import Path
            report = json.loads(next(Path(root).glob("audit_*.json")).read_text())
        tx.refresh_from_db()
        self.assertEqual(tx.classification, "gasto")
        self.assertEqual(len(report["changes"]), 1)
        self.assertTrue(any("negative_abono_requires_source" in w["reasons"] for w in report["warnings"]))

    def test_income_category_dry_run_and_future_import_with_confirmed_payer(self):
        from pathlib import Path
        from finance_app.core.services.classification import EARNED_INCOME_TYPE
        abono, _ = MovementType.objects.get_or_create(name="abono")
        income, _ = MovementType.objects.get_or_create(name=EARNED_INCOME_TYPE)
        income.keywords = ["empresa confirmada"]
        income.save()
        existing = Transaction.objects.create(user=self.user, amount=100, description="Transferencia PAGO NOMINA", movement_type=abono, classification="ingreso")
        with TemporaryDirectory() as root:
            call_command("categorize_income", output_dir=root)
            report = json.loads(next(Path(root).glob("income_categories_*.json")).read_text())
        self.assertEqual(report["counts"]["Transaction"], 1)
        existing.refresh_from_db()
        self.assertEqual(existing.movement_type, abono)
        rows = [{"date": "2026-01-01", "amount": 100, "description": "Abono Empresa Confirmada", "movement_type": "abono"},
                {"date": "2026-01-02", "amount": -20, "description": "Cargo por Recaudo con Debito Automatico", "movement_type": "abono"}]
        with patch("finance_app.core.services.pdf_parser.parse_pdf_file", return_value=(rows, [])):
            batch = stage_pdf_files(self.user, [Path("test.pdf")])
        staged = list(batch.staged_transactions.select_related("movement_type").order_by("id"))
        self.assertEqual(staged[0].movement_type.name, EARNED_INCOME_TYPE)
        self.assertEqual(staged[0].classification, "ingreso")
        self.assertEqual(staged[1].movement_type.name, "pago")
        self.assertEqual(staged[1].amount, Decimal("-20.00"))

    def test_pdf_import_uses_posted_date_for_oim_income(self):
        from pathlib import Path
        from finance_app.core.services.classification import EARNED_INCOME_TYPE
        rows = [
            {"date": "2021-12-31", "amount": 100, "description": "Abono CITIBANK", "movement_type": "abono"},
            {"date": "2022-01-01", "amount": 100, "description": "Abono CITIBANK", "movement_type": "abono"},
            {"date": "2026-01-01", "amount": 100, "description": "Abono ENGINEER SYSTEMS CORP SAS", "movement_type": "abono"},
        ]
        with patch("finance_app.core.services.pdf_parser.parse_pdf_file", return_value=(rows, [])):
            batch = stage_pdf_files(self.user, [Path("test.pdf")])
        names = list(batch.staged_transactions.order_by("id").values_list("movement_type__name", flat=True))
        self.assertEqual(names, [EARNED_INCOME_TYPE, "abono", EARNED_INCOME_TYPE])

    @patch("finance_app.core.services.pdf_parser.parse_pdf_file", return_value=([], [{"error": "failure"}]))
    def test_failed_import_rolls_back_batch(self, parser):
        from pathlib import Path
        with self.assertRaises(ValueError):
            stage_pdf_files(self.user, [Path("broken.pdf")])
        self.assertFalse(ImportBatch.objects.exists())


class ExtractionTests(SimpleTestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    def test_auto_uses_openai_when_native_text_cannot_be_verified(self):
        from PIL import Image
        from finance_app.core.services import pdf_parser
        with patch.object(pdf_parser, "PdfReader", side_effect=ValueError("scanned")), \
             patch.object(pdf_parser, "render_pdf_pages", return_value=([Image.new("RGB", (10, 10))], None, "test")), \
             patch.object(pdf_parser.pytesseract, "image_to_string", return_value="Bancolombia extracto 2026"), \
             patch.object(pdf_parser, "parse_with_openai", return_value=([{"description": "Abono", "source_page": 1}], {})) as extract:
            rows, debug = pdf_parser.parse_pdf_file("scan.pdf", extraction_method="auto")
        extract.assert_called_once()
        self.assertEqual(rows[0]["source_page"], 1)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test", "OPENAI_PDF_MODEL": "gpt-6-luna"})
    @patch("openai.OpenAI")
    def test_openai_processes_all_pages_and_keeps_account(self, client_cls):
        from PIL import Image
        from finance_app.core.services.pdf_parser import parse_with_openai
        row = {"date": "2026-01-01", "time": "", "description": "Abono", "reference": "", "amount": 100,
               "currency": "COP", "location": "", "channel": "", "bank": "Davivienda", "account_last4": "1234"}
        client_cls.return_value.responses.create.return_value = SimpleNamespace(status="completed", output_text=json.dumps({"transactions": [row]}))
        rows, debug = parse_with_openai([Image.new("RGB", (10, 10))] * 16, "Banco", "0000", 2026)
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[-1]["source_page"], 16)
        self.assertEqual(rows[-1]["account_last4"], "1234")
        self.assertEqual(debug["pages_sent"], 16)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    @patch("openai.OpenAI")
    def test_incomplete_response_is_not_imported(self, client_cls):
        from PIL import Image
        from finance_app.core.services.pdf_parser import parse_with_openai
        client_cls.return_value.responses.create.return_value = SimpleNamespace(status="incomplete")
        rows, debug = parse_with_openai([Image.new("RGB", (10, 10))], "Banco", "0000", 2026)
        self.assertEqual(rows, [])
        self.assertIn("Incomplete", debug["error"])
