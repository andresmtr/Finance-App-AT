from django.contrib import admin
from .models import MovementType, ImportBatch, StagedTransaction, Transaction

@admin.register(MovementType)
class MovementTypeAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "is_active")
    list_filter = ("is_active",)
    search_fields = ("name",)
    ordering = ("name",)

@admin.register(ImportBatch)
class ImportBatchAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "source_label", "created_at")
    list_filter = ("source_label", "created_at", "user")
    search_fields = ("source_label", "notes", "user__username", "user__email")
    date_hierarchy = "created_at"

@admin.register(StagedTransaction)
class StagedTransactionAdmin(admin.ModelAdmin):
    list_display = (
        "id", "user", "bank", "account_last4", "date", "amount", 
        "currency", "movement_type", "classification", "status", "created_at"
    )
    list_filter = ("status", "classification", "bank", "currency", "movement_type", "user")
    search_fields = (
        "description", "reference", "merchant", "bank", 
        "account_last4", "user__username", "user__email"
    )
    date_hierarchy = "date"
    readonly_fields = ("created_at",)

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = (
        "id", "user", "bank", "account_last4", "date", "amount", 
        "currency", "movement_type", "classification", "created_at"
    )
    list_filter = ("classification", "bank", "currency", "movement_type", "user")
    search_fields = (
        "description", "reference", "merchant", "bank", 
        "account_last4", "user__username", "user__email"
    )
    date_hierarchy = "date"
    readonly_fields = ("created_at",)
