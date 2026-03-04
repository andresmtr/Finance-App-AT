from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("transactions", "0004_populate_classification"),
    ]

    operations = [
        migrations.CreateModel(
            name="TelegramChatSession",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("chat_id", models.CharField(max_length=64, unique=True)),
                ("selected_date", models.DateField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
