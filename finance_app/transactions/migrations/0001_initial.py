from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="ImportBatch",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("source_label", models.CharField(blank=True, max_length=255)),
                ("notes", models.TextField(blank=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="import_batches",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="MovementType",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=64, unique=True)),
                ("keywords", models.JSONField(blank=True, default=list)),
                ("is_active", models.BooleanField(default=True)),
            ],
        ),
        migrations.CreateModel(
            name="Transaction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("bank", models.CharField(blank=True, max_length=128)),
                ("account_last4", models.CharField(blank=True, max_length=32)),
                ("date", models.DateField(blank=True, null=True)),
                ("time", models.TimeField(blank=True, null=True)),
                ("amount", models.DecimalField(decimal_places=2, max_digits=14)),
                ("currency", models.CharField(blank=True, max_length=8)),
                ("reference", models.CharField(blank=True, max_length=255)),
                ("merchant", models.CharField(blank=True, max_length=255)),
                ("location", models.CharField(blank=True, max_length=255)),
                ("channel", models.CharField(blank=True, max_length=255)),
                ("description", models.TextField(blank=True)),
                ("external_id", models.CharField(blank=True, max_length=64)),
                ("source_pdf", models.CharField(blank=True, max_length=255)),
                ("source_page", models.IntegerField(blank=True, null=True)),
                ("source_table", models.IntegerField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "movement_type",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="transactions.movementtype",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="transactions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="StagedTransaction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("bank", models.CharField(blank=True, max_length=128)),
                ("account_last4", models.CharField(blank=True, max_length=32)),
                ("date", models.DateField(blank=True, null=True)),
                ("time", models.TimeField(blank=True, null=True)),
                ("amount", models.DecimalField(blank=True, decimal_places=2, max_digits=14, null=True)),
                ("currency", models.CharField(blank=True, max_length=8)),
                ("reference", models.CharField(blank=True, max_length=255)),
                ("merchant", models.CharField(blank=True, max_length=255)),
                ("location", models.CharField(blank=True, max_length=255)),
                ("channel", models.CharField(blank=True, max_length=255)),
                ("description", models.TextField(blank=True)),
                ("external_id", models.CharField(blank=True, max_length=64)),
                ("source_pdf", models.CharField(blank=True, max_length=255)),
                ("source_page", models.IntegerField(blank=True, null=True)),
                ("source_table", models.IntegerField(blank=True, null=True)),
                (
                    "status",
                    models.CharField(
                        choices=[("pending", "Pendiente"), ("approved", "Aprobada"), ("skipped", "Omitida")],
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "batch",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="staged_transactions",
                        to="transactions.importbatch",
                    ),
                ),
                (
                    "movement_type",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="transactions.movementtype",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="staged_transactions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
