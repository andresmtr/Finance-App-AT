import logging
import os
import shutil
from datetime import datetime

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpRequest, HttpResponse, FileResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from finance_app.core.services.ingestion import update_consolidated

LOGGER = logging.getLogger(__name__)


def health(_request: HttpRequest) -> HttpResponse:
    return HttpResponse("ok", content_type="text/plain")


def signup(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("/")
    else:
        form = UserCreationForm()
    return render(request, "registration/signup.html", {"form": form})


@require_POST
@login_required
def update_data(request: HttpRequest) -> HttpResponse:
    try:
        result = update_consolidated()
        messages.success(
            request,
            f"Actualizacion completa: {result['files']} archivos, {result['rows']} filas.",
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Fallo la actualizacion: %s", exc)
        messages.error(request, f"Error al actualizar datos: {exc}")
    return redirect(request.META.get("HTTP_REFERER", "/"))


@login_required
@user_passes_test(lambda u: u.is_superuser)
def download_backup(request: HttpRequest) -> HttpResponse:
    db_path = settings.DATABASES["default"]["NAME"]
    if not os.path.exists(db_path):
        messages.error(request, "Archivo de base de datos no encontrado.")
        return redirect("/")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backup_finance_{timestamp}.sqlite3"
    
    response = FileResponse(open(db_path, "rb"), content_type="application/x-sqlite3")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@login_required
@user_passes_test(lambda u: u.is_superuser)
@require_POST
def restore_backup(request: HttpRequest) -> HttpResponse:
    if "backup_file" not in request.FILES:
        messages.error(request, "No se subió ningún archivo.")
        return redirect("/")
    
    backup_file = request.FILES["backup_file"]
    if not backup_file.name.endswith(".sqlite3"):
        messages.error(request, "El archivo debe tener extensión .sqlite3")
        return redirect("/")
    
    db_path = settings.DATABASES["default"]["NAME"]
    
    # Crear un backup temporal del actual antes de sobreescribir
    shutil.copy2(db_path, f"{db_path}.bak")
    
    try:
        with open(db_path, "wb+") as destination:
            for chunk in backup_file.chunks():
                destination.write(chunk)
        messages.success(request, "Base de datos restaurada con éxito. Es posible que debas volver a iniciar sesión.")
    except Exception as e:
        LOGGER.exception("Error restaurando backup: %s", e)
        shutil.move(f"{db_path}.bak", db_path) # Restaurar original si falla
        messages.error(request, f"Error al restaurar: {e}")
        
    return redirect("/")
