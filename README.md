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
```

`PDF_DEBUG=1` activa logs de parsing similares al notebook.

## Desarrollo local (opcional)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
