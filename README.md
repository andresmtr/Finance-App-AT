# Finanzas Personales (Django + Samba)

App web para consolidar CSV desde un recurso Samba y visualizar KPIs, graficas y transacciones.

## Requisitos
- Docker + Docker Compose

## Configuracion
1) Copia `.env.example` a `.env` y completa las variables:
```
DJANGO_SECRET_KEY=...
DEBUG=true
ALLOWED_HOSTS=*
SAMBA_HOST=192.168.20.24
SAMBA_USERNAME=samba
SAMBA_PASSWORD=...
SAMBA_REMOTE_PATH=/backup/n8n/datos_bk
```

Nota: `SAMBA_REMOTE_PATH` se interpreta como `/share/subcarpeta`. En el ejemplo anterior el share es `backup` y la carpeta es `n8n/datos_bk`.

## Levantar con Docker
```
docker compose up --build
```

La app queda disponible en `http://localhost:8000`.

## Uso
- Dashboard: KPIs y graficas principales.
- Transacciones: tabla con filtros y export CSV.
- Boton "Actualizar": descarga los CSV desde Samba, consolida y guarda en `data/consolidated.csv`.
- Inicio de sesion: requiere las credenciales del superusuario Django.

## Estructura de datos
- Se normalizan columnas y se toleran CSV con columnas faltantes.
- Deduplicacion: usa `id` si existe; si no, crea hash de `(bank, account_last4, date, time, amount, reference)`.

## Troubleshooting
- **Permisos Samba**: verifica usuario/clave y que el share exponga la ruta correcta.
- **Samba offline**: el boton "Actualizar" mostrara el error en pantalla.
- **Encoding CSV**: se usa `encoding_errors=replace` y `on_bad_lines=skip` para evitar bloqueos.
- **Separador CSV**: se intenta inferir el separador automaticamente.

## Desarrollo local (opcional)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```
