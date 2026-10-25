import os
from django.conf import settings
from google import genai
from openai import OpenAI


class AIProviderError(Exception):
    """Raised when the configured AI provider cannot be used."""


class AIProvider:
    SUPPORTED_PROVIDERS = {"openai", "gemini"}

    def __init__(self, provider=None):
        self.provider = (provider or os.getenv("AI_DEFAULT_PROVIDER", "openai")).lower()
        self.openai_key = getattr(settings, "OPENAI_API_KEY", "")
        self.gemini_key = getattr(settings, "GEMINI_API_KEY", "")

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise AIProviderError(f"Proveedor no soportado: {self.provider}")

        if self.provider == "openai":
            if not self.openai_key:
                raise AIProviderError("OPENAI_API_KEY no esta configurada.")
            self.model = getattr(settings, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
            self.client = OpenAI(api_key=self.openai_key)
        elif self.provider == "gemini":
            if not self.gemini_key:
                raise AIProviderError("GEMINI_API_KEY no esta configurada.")
            self.model = getattr(settings, "GEMINI_CHAT_MODEL", "gemini-3-flash-preview")
            self.client = genai.Client(api_key=self.gemini_key)

    def get_response(self, system_prompt, user_query):
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content
        elif self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                config={
                    "system_instruction": system_prompt
                },
                contents=user_query
            )
            return response.text
        raise AIProviderError(f"Proveedor no soportado: {self.provider}")


def get_db_schema_context(user_id=None):
    """
    Returns a description of the database schema for the LLM.
    """
    user_filter = ""
    if user_id is not None:
        user_filter = f"""
    Authenticated user:
    - The current user_id is {user_id}.
    - Every query that reads transactions_transaction MUST filter by user_id = {user_id}.
    """

    schema = """
    You are an AI assistant for a Personal Finance application. 
    You have access to information about the following database tables (read-only):
    
    1. Table: transactions_transaction
       - id: primary key
       - user_id: ID of the user
       - bank: name of the bank
       - account_last4: last 4 digits of the account
       - date: date of the transaction (YYYY-MM-DD)
       - amount: decimal amount. Values can be positive or negative; use classification and ABS(amount) for expense totals.
       - currency: e.g., COP, USD
       - movement_type_id: ID of the category (MovementType)
       - merchant: name of the merchant
       - description: description of the transaction
       - classification: 'ingreso' (income), 'gasto' (expense), 'cdt', 'bolsillo'
       
    2. Table: transactions_movementtype
       - id: primary key
       - name: category name (e.g., Restaurantes, Transporte, etc.)
       
    Instructions:
    - Generate one SQLite compatible query.
    - IMPORTANT: Only perform SELECT queries.
    - Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, DETACH, VACUUM, or comments.
    - Prefer joins to transactions_movementtype when the user asks for categories.
    - Use ABS(amount) when calculating expense totals.
    - Return ONLY the SQL query. No markdown, no comments.
    """
    return f"{schema}\n{user_filter}"
