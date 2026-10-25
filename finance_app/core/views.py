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


def _backup_source_path():
    from pathlib import Path
    return Path(settings.DATABASES["default"]["NAME"])


@login_required
@user_passes_test(lambda u: u.is_superuser)
def download_backup(request: HttpRequest) -> HttpResponse:
    import sqlite3
    import tempfile
    from contextlib import closing
    from pathlib import Path
    db_path = _backup_source_path()
    if not db_path.is_file():
        messages.error(request, "Archivo de base de datos no encontrado.")
        return redirect("/")
    temporary = tempfile.NamedTemporaryFile(suffix=".sqlite3")
    try:
        with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as source, closing(sqlite3.connect(temporary.name)) as target:
            source.backup(target)
        temporary.seek(0)
        response = FileResponse(temporary, content_type="application/x-sqlite3", as_attachment=True,
                                filename=f"backup_finance_{datetime.now():%Y%m%d_%H%M%S}.sqlite3")
        response["Cache-Control"] = "private, no-store"
        return response
    except Exception:
        temporary.close()
        raise



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


@login_required
@user_passes_test(lambda u: u.is_superuser)
def download_full_backup(request):
    """Consistent database snapshot and referenced PDFs, streamed as a private archive."""
    import json
    import sqlite3
    import tempfile
    import zipfile
    from contextlib import closing
    from pathlib import Path
    from finance_app.transactions.models import CardStatement, StagedTransaction, Transaction

    root=Path(settings.MEDIA_ROOT).resolve()
    archive=tempfile.SpooledTemporaryFile(max_size=16*1024*1024)
    missing=[];included=[]
    try:
        with tempfile.NamedTemporaryFile(suffix='.sqlite3') as snapshot:
            source_path=_backup_source_path().resolve()
            with closing(sqlite3.connect(source_path.as_uri()+'?mode=ro',uri=True)) as source,closing(sqlite3.connect(snapshot.name)) as target:source.backup(target)
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(snapshot.name,'db.sqlite3')
                references={('pdfs',name) for name in Transaction.objects.exclude(source_pdf='').values_list('source_pdf',flat=True)}
                references|={('pdfs',name) for name in StagedTransaction.objects.exclude(source_pdf='').values_list('source_pdf',flat=True)}
                references|={('cards',name) for name in CardStatement.objects.values_list('source_pdf',flat=True)}
                for folder,name in sorted(references):
                    path=(root/folder/name).resolve()
                    archive_name=f'uploads/{folder}/{name}'
                    if path.parent!=(root/folder).resolve() or not path.is_file():missing.append(archive_name);continue
                    bundle.write(path,archive_name);included.append(archive_name)
                bundle.writestr('manifest.json',json.dumps({'created_at':datetime.now().isoformat(),'database':'db.sqlite3','pdfs':included,'missing_pdfs':missing},ensure_ascii=False,indent=2))
        archive.seek(0)
        response=FileResponse(archive,content_type='application/zip',as_attachment=True,filename=f'finance_completo_{datetime.now():%Y%m%d_%H%M%S}.zip')
        response['Cache-Control']='private, no-store';return response
    except Exception:
        archive.close();raise
