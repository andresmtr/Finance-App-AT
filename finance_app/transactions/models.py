from django.conf import settings
from django.db import models


class MovementType(models.Model):
    name = models.CharField(max_length=64, unique=True)
    keywords = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return self.name


class ImportBatch(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="import_batches",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    source_label = models.CharField(max_length=255, blank=True)
    notes = models.TextField(blank=True)

    def __str__(self) -> str:
        return f"{self.user} - {self.created_at:%Y-%m-%d %H:%M}"


class StagedTransaction(models.Model):
    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_SKIPPED = "skipped"
    STATUS_DELETED = "deleted"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pendiente"),
        (STATUS_APPROVED, "Aprobada"),
        (STATUS_SKIPPED, "Omitida"),
        (STATUS_DELETED, "Eliminada"),
    ]
    
    CLASSIFICATION_CHOICES = [
        ('ingreso', 'Ingreso'),
        ('gasto', 'Gasto'),
        ('cdt', 'CDT'),
        ('bolsillo', 'Bolsillo'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="staged_transactions",
    )
    batch = models.ForeignKey(
        ImportBatch,
        on_delete=models.CASCADE,
        related_name="staged_transactions",
    )
    bank = models.CharField(max_length=128, blank=True)
    account_last4 = models.CharField(max_length=32, blank=True)
    date = models.DateField(null=True, blank=True)
    time = models.TimeField(null=True, blank=True)
    amount = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    currency = models.CharField(max_length=8, blank=True)
    movement_type = models.ForeignKey(
        MovementType,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    reference = models.CharField(max_length=255, blank=True)
    merchant = models.CharField(max_length=255, blank=True)
    location = models.CharField(max_length=255, blank=True)
    channel = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    external_id = models.CharField(max_length=64, blank=True)
    source_pdf = models.CharField(max_length=255, blank=True)
    source_page = models.IntegerField(null=True, blank=True)
    source_table = models.IntegerField(null=True, blank=True)
    status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )
    classification = models.CharField(
        max_length=16,
        choices=CLASSIFICATION_CHOICES,
        default='gasto',
    )
    created_at = models.DateTimeField(auto_now_add=True)


class Transaction(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="transactions",
    )
    bank = models.CharField(max_length=128, blank=True)
    account_last4 = models.CharField(max_length=32, blank=True)
    date = models.DateField(null=True, blank=True)
    time = models.TimeField(null=True, blank=True)
    amount = models.DecimalField(max_digits=14, decimal_places=2)
    currency = models.CharField(max_length=8, blank=True)
    movement_type = models.ForeignKey(
        MovementType,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    reference = models.CharField(max_length=255, blank=True)
    merchant = models.CharField(max_length=255, blank=True)
    location = models.CharField(max_length=255, blank=True)
    channel = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    external_id = models.CharField(max_length=64, blank=True)
    source_pdf = models.CharField(max_length=255, blank=True)
    source_page = models.IntegerField(null=True, blank=True)
    source_table = models.IntegerField(null=True, blank=True)
    classification = models.CharField(
        max_length=16,
        choices=StagedTransaction.CLASSIFICATION_CHOICES,
        default='gasto',
    )
    created_at = models.DateTimeField(auto_now_add=True)


class TelegramChatSession(models.Model):
    chat_id = models.CharField(max_length=64, unique=True)
    selected_date = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Telegram chat {self.chat_id} ({self.selected_date or 'sin_fecha'})"


class CreditCard(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="credit_cards")
    bank = models.CharField(max_length=128)
    last4 = models.CharField(max_length=4)
    label = models.CharField(max_length=128, blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "bank", "last4"], name="unique_user_credit_card")]

    def __str__(self):
        return self.label or f"{self.bank} · {self.last4}"


class CardStatement(models.Model):
    card = models.ForeignKey(CreditCard, on_delete=models.CASCADE, related_name="statements")
    period_start = models.DateField()
    period_end = models.DateField()
    due_date = models.DateField(null=True, blank=True)
    currency = models.CharField(max_length=8, default="COP")
    opening_balance = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    closing_balance = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    purchases_total = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    payments_total = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    source_pdf = models.CharField(max_length=255)
    file_hash = models.CharField(max_length=64)
    validation_notes = models.JSONField(default=list, blank=True)
    reviewed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["card", "period_start", "period_end", "currency"], name="unique_card_statement_period")]
        ordering = ["-period_end", "-pk"]


class CardMovement(models.Model):
    KINDS = [("purchase", "Compra"), ("refund", "Devolución"), ("payment", "Pago de tarjeta"),
             ("interest", "Intereses"), ("fee", "Comisión / cuota"), ("tax", "Impuesto"),
             ("advance", "Avance"), ("adjustment", "Ajuste")]
    CATEGORIES = [("restaurantes", "Restaurantes"), ("mercado", "Mercado y supermercado"),
                  ("compras", "Compras"), ("tecnologia", "Tecnología y suscripciones"),
                  ("transporte", "Transporte y combustible"), ("viajes", "Viajes y alojamiento"),
                  ("salud", "Salud"), ("entretenimiento", "Entretenimiento"),
                  ("servicios", "Servicios"), ("financieros", "Costos financieros"),
                  ("otros", "Otros / por clasificar")]
    card = models.ForeignKey(CreditCard, on_delete=models.CASCADE, related_name="movements")
    date = models.DateField()
    kind = models.CharField(max_length=16, choices=KINDS)
    description = models.TextField()
    merchant = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=32, choices=CATEGORIES, default="otros")
    # Original purchase principal, positive charges and negative payments/refunds.
    # A repeated installment line points to the same movement in subsequent statements.
    amount = models.DecimalField(max_digits=16, decimal_places=2)
    currency = models.CharField(max_length=8, default="COP")
    reference = models.CharField(max_length=128, blank=True)
    fingerprint = models.CharField(max_length=64)
    manually_classified = models.BooleanField(default=False)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["card", "fingerprint"], name="unique_card_movement")]
        ordering = ["-date", "-pk"]


class CardStatementLine(models.Model):
    statement = models.ForeignKey(CardStatement, on_delete=models.CASCADE, related_name="lines")
    movement = models.ForeignKey(CardMovement, on_delete=models.CASCADE, related_name="statement_lines")
    billed_amount = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    remaining_balance = models.DecimalField(max_digits=16, decimal_places=2, null=True, blank=True)
    installment = models.PositiveIntegerField(null=True, blank=True)
    installments = models.PositiveIntegerField(null=True, blank=True)
    source_page = models.PositiveIntegerField(default=1)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["statement", "movement"], name="unique_card_statement_line")]


class CardPaymentAllocation(models.Model):
    bank_transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name="card_allocations")
    card_payment = models.ForeignKey(CardMovement, on_delete=models.CASCADE, related_name="allocations")
    amount = models.DecimalField(max_digits=16, decimal_places=2)
    method = models.CharField(max_length=16, choices=[("auto", "Automática"), ("manual", "Manual")], default="manual")
    notes = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["bank_transaction", "card_payment"], name="unique_card_payment_allocation"),
            models.CheckConstraint(check=models.Q(amount__gt=0), name="positive_card_allocation"),
        ]


class CardExpenseAllocation(models.Model):
    bank_transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name="expense_allocations")
    card_expense = models.ForeignKey(CardMovement, on_delete=models.CASCADE, related_name="expense_allocations")
    amount = models.DecimalField(max_digits=16, decimal_places=2)
    notes = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["bank_transaction", "card_expense"], name="unique_card_expense_allocation"),
            models.CheckConstraint(check=models.Q(amount__gt=0), name="positive_card_expense_allocation"),
        ]
