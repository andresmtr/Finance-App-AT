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
    created_at = models.DateTimeField(auto_now_add=True)
