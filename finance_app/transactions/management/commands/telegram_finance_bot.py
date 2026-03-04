from __future__ import annotations

import time
import urllib.error

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from finance_app.transactions.services.telegram_finance import (
    FinanceAssistant,
    TelegramBotClient,
    build_reply_for_chat_message,
    build_help_text,
)


class Command(BaseCommand):
    help = "Run a Telegram bot that answers natural-language questions from SQLite transactions."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Process one polling cycle and exit.")
        parser.add_argument("--offset", type=int, default=0, help="Initial Telegram update offset.")
        parser.add_argument(
            "--user-id",
            type=int,
            default=None,
            help="Django user_id used for queries (defaults to TELEGRAM_DEFAULT_USER_ID).",
        )

    def handle(self, *args, **options):
        token = settings.TELEGRAM_BOT_TOKEN.strip()
        if not token:
            raise CommandError("TELEGRAM_BOT_TOKEN is empty in .env.")

        default_user_id = options["user_id"] or settings.TELEGRAM_DEFAULT_USER_ID
        allowed_ids = {str(x) for x in settings.TELEGRAM_ALLOWED_CHAT_IDS}

        self.stdout.write(self.style.SUCCESS("Starting Telegram finance bot..."))
        self.stdout.write(f"Default user_id: {default_user_id}")
        if allowed_ids:
            self.stdout.write(f"Allowed chat IDs: {', '.join(sorted(allowed_ids))}")

        bot = TelegramBotClient(token=token, timeout=settings.TELEGRAM_POLL_TIMEOUT)
        assistant = FinanceAssistant()
        offset = int(options["offset"])
        run_once = bool(options["once"])

        while True:
            try:
                updates = bot.get_updates(offset=offset, timeout=settings.TELEGRAM_POLL_TIMEOUT)
                for update in updates:
                    offset = max(offset, int(update["update_id"]) + 1)
                    message = update.get("message") or {}
                    text = (message.get("text") or "").strip()
                    chat = message.get("chat") or {}
                    chat_id = chat.get("id")
                    if not text or chat_id is None:
                        continue

                    if allowed_ids and str(chat_id) not in allowed_ids:
                        continue

                    if text.startswith("/start") or text.startswith("/help") or text.startswith("/ayuda"):
                        bot.send_message(chat_id=chat_id, text=build_help_text())
                        continue

                    answer = build_reply_for_chat_message(
                        chat_id=str(chat_id),
                        text=text,
                        assistant=assistant,
                        user_id=default_user_id,
                    )
                    bot.send_message(chat_id=chat_id, text=answer[:4096])
            except urllib.error.URLError as exc:
                self.stderr.write(f"Telegram connection error: {exc}")
                time.sleep(3)
            except Exception as exc:  # pragma: no cover
                self.stderr.write(f"Unexpected error: {exc}")
                time.sleep(3)

            if run_once:
                break
