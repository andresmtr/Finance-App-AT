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

Reglas confirmadas del pagador: los ingresos de ENGINEER SYSTEMS CORP SAS (incluido su NIT 9009495364) se clasifican como `pago de salarios u honorarios`. Los abonos de CITIBANK/CIBAN con fecha de movimiento hasta el 31 de diciembre de 2021 inclusive se clasifican como `pago de salarios u honorarios`. Los movimientos posteriores o sin fecha no reciben automáticamente esa categoría de OIM. Los débitos, reembolsos, intereses y movimientos de bolsillos conservan sus reglas específicas.


### Datos privados y despliegue

Git y el contexto de construcción de Docker excluyen las bases de datos, respaldos, PDF, exportaciones, notebooks, credenciales y las carpetas `data/` y `uploads/`. Los datos se mantienen localmente y se trasladan al servidor por un canal privado, separados del despliegue del código. Para conservar el botón de extracto original al restaurar la base, también se debe copiar `uploads/pdfs/` al almacenamiento privado del servidor.

Eliminar un archivo del índice o añadirlo a `.gitignore` no lo elimina de los commits anteriores. El historial publicado que contenga datos privados requiere una limpieza específica del historial remoto.


### Tarjetas, consumos y conciliación

La sección `/cards/` mantiene los consumos de tarjeta separados de los movimientos de cuentas. Una compra a cuotas se registra una sola vez por su valor original; cada extracto conserva el importe a pagar, saldo pendiente y cuotas. Los pagos reducen la deuda y se cruzan con las salidas bancarias sin generar un gasto adicional en el flujo de caja.

- Carga nuevos PDF desde **Tarjetas y consumos**. La lectura nativa admite extractos de Davivienda y Banco de Bogotá; los escaneos utilizan el modelo PDF configurado. Los archivos y sus contraseñas no se incluyen en el repositorio; las contraseñas del formulario no se guardan.
- Los cruces automáticos requieren importe exacto, moneda coincidente, diferencia de hasta siete días y un candidato único en ambos lados; se respeta la tarjeta destino cuando aparece en la descripción. Las dudas y saldos sin cruce quedan visibles.
- En **Conciliar** puedes marcar varias compras o costos para asociarlos a un pago, incluso de otro mes. Se admiten importes parciales y se impide exceder el pago o el gasto. Este desglose manual se conserva junto al cruce entre el pago de cuenta y el abono del extracto.
- El Excel de tarjetas incluye consumos, abonos, pagos de cuenta, conciliaciones, gastos asociados, extractos, detalle y alcance de filtros. Las compras se filtran por fecha de compra y los extractos por fecha de corte. El Excel de cuentas conserva los filtros y exporta un archivo válido incluso sin resultados.
- El respaldo SQLite usa una instantánea consistente. **Descargar respaldo completo (base + PDF)** genera un ZIP privado con `db.sqlite3`, los PDF referenciados y `manifest.json`. Para el servidor, conserva las rutas `uploads/pdfs/` y `uploads/cards/`. El formulario de restauración existente recibe el archivo `.sqlite3`, no el ZIP.
- Después de desplegar código nuevo o restaurar una base anterior, ejecuta `python manage.py migrate --noinput`. Las migraciones de tarjetas crean las tablas nuevas sin reemplazar los movimientos de cuentas.

Para importaciones masivas locales: `python manage.py import_card_statements --manifest ruta_privada.json --user-id ID`. La contraseña se lee de `CARD_PDF_PASSWORD`; no se almacena en el manifiesto ni en los informes. `--refresh` vuelve a leer los períodos existentes, conserva las clasificaciones y cruces de movimientos que no cambian, y retira los cruces de filas corregidas.


Las reglas de consumo agrupan droguerías/farmacias en **Salud**, OXXO en **Supermercados**, facturas de agua, luz, gas, celular, planes de datos e internet en **Servicios públicos**, y compras del hogar en **Gastos de casa**. `python manage.py categorize_card_expenses --apply` actualiza únicamente esas coincidencias y guarda un respaldo previo. Los pagos de crédito hipotecario se muestran en el bloque de vivienda y en su propia hoja del Excel, conservando los importes ya registrados en las cuentas. Para identificar préstamos cuyo extracto solo dice “Cuota Cartera”, configura `HOUSING_CREDIT_LAST4` con los créditos confirmados como vivienda. Esa identificación permanece en `.env`, fuera de Git.

La categoría **Educación** reconoce GIMNASIO MODERNO CASTILLA (incluido el nombre truncado CASTIL), UNAD, Preply y Udemy, también cuando el cargo aparece a través de PayPal. Los pagos directos desde cuentas se muestran por separado en el bloque de Educación y en la hoja `Educacion cuentas` del Excel. Los intereses y comisiones de las tarjetas conservan su categoría financiera.
