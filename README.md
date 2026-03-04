# Finanzas Personales (Django + OCR PDF)

App web para importar extractos bancarios en PDF, revisar los movimientos en una lista editable y guardar transacciones por usuario. Incluye dashboard de KPIs, filtros y export CSV.

## Requisitos
- Docker + Docker Compose
- Poppler y Tesseract dentro del contenedor (incluidos en el Dockerfile)

## Levantar con Docker
```
docker compose up --build
```

La app queda disponible en `http://localhost:8002`.

## Uso
- Registro e inicio de sesion con Django.
- Importar PDFs: carga uno o varios PDFs, procesa OCR y genera movimientos en estado pendiente.
- Revision por lote: lista editable para ajustar, aprobar, omitir o eliminar movimientos (sin borrado fisico).
- Transacciones: tabla con filtros y export CSV.
- Dashboard: KPIs y graficas basadas en la base de datos.
- Idiomas: selector en la barra superior (es/en).

## Flujo de datos
- OCR con Tesseract + deteccion de tablas con Table Transformer.
- Parsing de montos con debito/credito y fallback por texto.
- Los PDFs se eliminan despues del staging.
- Datos persistidos en SQLite por usuario.

## Configuracion opcional
Variables en `.env`:
```
DJANGO_SECRET_KEY=...
DEBUG=true
ALLOWED_HOSTS=*
PDF_DEBUG=1
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_CHAT_IDS=793260423
TELEGRAM_DEFAULT_USER_ID=1
TELEGRAM_POLL_TIMEOUT=30
TELEGRAM_WEBHOOK_SECRET=...
TELEGRAM_WEBHOOK_URL=https://tu-dominio.com/telegram/webhook/
```

`PDF_DEBUG=1` activa logs de parsing similares al notebook.

## Bot de Telegram (preguntas en lenguaje natural)
Conecta Telegram directo a Django/SQLite para consultar gastos/ingresos sin n8n.

1) Configura un token nuevo en `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_ALLOWED_CHAT_IDS=793260423
TELEGRAM_WEBHOOK_SECRET=mi-secreto-largo
TELEGRAM_WEBHOOK_URL=https://tu-dominio.com/telegram/webhook/
```

2) Registra el webhook:
```bash
python manage.py telegram_set_webhook --info
```

3) Escribe al bot en Telegram.
Flujo conversacional (como tu n8n):
- Envia fecha: `2026-01-12` (o `hoy`, `ayer`)
- Luego pregunta: `cuanto gaste`, `top 5 comercios`, `cuantas transacciones`
- Cierra sesion: `finalizar`

Notas:
- Ruta webhook en la app: `/telegram/webhook/`
- Si no quieres webhook, puedes usar polling:
```bash
python manage.py telegram_finance_bot
```

Preguntas ejemplo:
- `2026-01-12`
- `cuanto gaste`
- `top 5 comercios`
- `ultimas 10 transacciones`
- `finalizar`

## Desarrollo local (opcional)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
