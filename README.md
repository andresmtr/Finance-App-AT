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
GEMINI_API_KEY=...
GEMINI_CHAT_MODEL=gemini-3-flash-preview
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


### Extracción y conciliación de extractos

- `OPENAI_PDF_MODEL=gpt-6-luna` configura el modelo de PDF independientemente del chat.
- El modo automático intenta leer texto nativo y comprobar los saldos. Si no puede validarlo, utiliza OpenAI cuando hay una clave configurada. La extracción procesa todas las páginas con salida JSON estructurada; las secciones de tarjetas están excluidas.
- Los PDF originales se conservan en `uploads/pdfs/`. El botón de PDF en cada movimiento requiere autenticación y comprueba su propietario.
- Los intereses de CDT son ingresos separados del capital. Los traslados entre cuenta y bolsillos, incluidos sus rendimientos ya abonados, no afectan al ingreso ni al gasto operativo.
- `python manage.py audit_finances` genera un informe privado en `data/audits/`. `--apply` guarda una copia de SQLite antes de corregir agrupaciones inequívocas; no cambia signos ni elimina posibles duplicados.
- Las importaciones pendientes se pueden abrir desde `/imports/`; sus movimientos no afectan los totales hasta aprobarlos.


Los ingresos con conceptos de nómina, honorarios, contratista, prima o liquidación laboral se agrupan como `pago de salarios u honorarios`. La detección exige un importe positivo y conserva la descripción bancaria original. `python manage.py categorize_income` permite revisar los cambios y `--apply` aplicarlos con copia de seguridad. `--income-keyword "nombre del pagador confirmado"` guarda frases confirmadas en las palabras clave del tipo de movimiento para reconocer futuras importaciones; el nombre de un banco, por sí solo, no se considera prueba de un pago laboral.

Reglas confirmadas del pagador: los ingresos de ENGINEER SYSTEMS CORP SAS (incluido su NIT 9009495364) se clasifican como `pago de salarios`. Los abonos de CITIBANK/CIBAN con fecha de movimiento hasta el 31 de diciembre de 2021 inclusive se clasifican como `honorarios OIM`. Los movimientos posteriores o sin fecha no reciben automáticamente esa categoría de OIM. Los débitos, reembolsos, intereses y movimientos de bolsillos conservan sus reglas específicas.


### Datos privados y despliegue

Git y el contexto de construcción de Docker excluyen las bases de datos, respaldos, PDF, exportaciones, notebooks, credenciales y las carpetas `data/` y `uploads/`. Los datos se mantienen localmente y se trasladan al servidor por un canal privado, separados del despliegue del código. Para conservar el botón de extracto original al restaurar la base, también se debe copiar `uploads/pdfs/` al almacenamiento privado del servidor.

Eliminar un archivo del índice o añadirlo a `.gitignore` no lo elimina de los commits anteriores. El historial publicado que contenga datos privados requiere una limpieza específica del historial remoto.
