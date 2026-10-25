import json
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from finance_app.transactions.models import Transaction


class AIChatTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="andres",
            password="test-pass",
        )
        self.client.force_login(self.user)

    def test_chat_template_uses_browser_json_api(self):
        response = self.client.get(reverse("ai_chat:chat"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "JSON.stringify")
        self.assertNotContains(response, "json.stringify")
        self.assertContains(response, "Consulta local")

    def test_ask_ai_filters_by_description_text(self):
        Transaction.objects.create(
            user=self.user,
            bank="Test Bank",
            amount=Decimal("-25000.00"),
            merchant="",
            description="Compra Cafe Test",
            classification="gasto",
        )

        response = self.client.post(
            reverse("ai_chat:ask_ai"),
            data=json.dumps({"query": "cuanto gaste en cafe"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "total_gastos")
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["payload"]["filters"]["search_text"], "cafe")
        self.assertEqual(payload["payload"]["total"], 25000.0)

    def test_ask_ai_top_descriptions_uses_description_when_merchant_is_empty(self):
        Transaction.objects.create(
            user=self.user,
            amount=Decimal("-10000.00"),
            merchant="",
            description="Compra TECNIPAGOS S A",
            classification="gasto",
        )
        Transaction.objects.create(
            user=self.user,
            amount=Decimal("-15000.00"),
            merchant="",
            description="Compra TECNIPAGOS S A",
            classification="gasto",
        )

        response = self.client.post(
            reverse("ai_chat:ask_ai"),
            data=json.dumps({"query": "top descripciones de gastos"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "top_descripciones")
        self.assertEqual(payload["payload"]["top_descriptions"][0]["descripcion"], "Compra TECNIPAGOS S A")
        self.assertEqual(payload["payload"]["top_descriptions"][0]["total"], 25000.0)

    def test_ask_ai_marks_out_of_domain_question_invalid(self):
        response = self.client.post(
            reverse("ai_chat:ask_ai"),
            data=json.dumps({"query": "quien gano el partido"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["valid"])

    def test_ask_ai_rejects_empty_query(self):
        response = self.client.post(
            reverse("ai_chat:ask_ai"),
            data=json.dumps({"query": "", "provider": "openai"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("pregunta", response.json()["error"].lower())
