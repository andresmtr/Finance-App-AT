from django.db import migrations


def seed_movement_types(apps, _schema_editor):
    MovementType = apps.get_model("transactions", "MovementType")
    rules = [
        ("cdt", ["cdt", "certificado de deposito", "certificado de depósito", "título", "titulo"]),
        ("intereses", ["interes", "interés", "rendimiento", "rendimientos"]),
        ("impuesto", ["gmf", "4x1000", "impuesto", "retencion", "retención", "iva"]),
        ("retiro", ["retiro", "atm", "cajero", "withdrawal"]),
        ("transferencia", ["transferencia", "transf", "pse", "ach", "envio", "envío"]),
        ("pago", ["pago", "cuota", "tarj", "tarjeta", "credito", "crédito", "servicio", "manejo"]),
        ("compra", ["compra", "pos", "datáfono", "datafono", "comercio", "apple.com", "bill", "supermercado", "mercado"]),
        ("abono", ["abono", "consignacion", "consignación", "deposito", "depósito", "ingreso", "recaudo"]),
    ]
    for name, keywords in rules:
        MovementType.objects.update_or_create(
            name=name,
            defaults={"keywords": keywords, "is_active": True},
        )


class Migration(migrations.Migration):
    dependencies = [
        ("transactions", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(seed_movement_types, migrations.RunPython.noop),
    ]
