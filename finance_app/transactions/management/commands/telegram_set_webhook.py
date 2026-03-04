from __future__ import annotations

import json
import urllib.parse
import urllib.request

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Configure Telegram webhook for direct integration with this Django app."

    def add_arguments(self, parser):
        parser.add_argument("--url", default="", help="Public webhook URL. Fallback: TELEGRAM_WEBHOOK_URL from .env")
        parser.add_argument("--delete", action="store_true", help="Delete current webhook")
        parser.add_argument("--info", action="store_true", help="Get webhook info after update/delete")

    def handle(self, *args, **options):
        token = settings.TELEGRAM_BOT_TOKEN.strip()
        if not token:
            raise CommandError("TELEGRAM_BOT_TOKEN is empty in .env.")

        if options["delete"]:
            result = self._call(token, "deleteWebhook", {})
            self.stdout.write(self.style.SUCCESS(f"deleteWebhook: {json.dumps(result, ensure_ascii=True)}"))
        else:
            url = (options["url"] or settings.TELEGRAM_WEBHOOK_URL).strip()
            if not url:
                raise CommandError("Webhook URL is required. Use --url or TELEGRAM_WEBHOOK_URL in .env.")
            payload = {"url": url}
            secret = settings.TELEGRAM_WEBHOOK_SECRET.strip()
            if secret:
                payload["secret_token"] = secret
            result = self._call(token, "setWebhook", payload)
            self.stdout.write(self.style.SUCCESS(f"setWebhook: {json.dumps(result, ensure_ascii=True)}"))

        if options["info"]:
            info = self._call(token, "getWebhookInfo", {})
            self.stdout.write(self.style.SUCCESS(f"getWebhookInfo: {json.dumps(info, ensure_ascii=True)}"))

    def _call(self, token: str, method: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url=f"https://api.telegram.org/bot{token}/{method}",
            data=urllib.parse.urlencode(payload).encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not body.get("ok"):
            raise CommandError(f"Telegram API error in {method}: {body}")
        return body.get("result", {})
