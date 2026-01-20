import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.views.decorators.http import require_POST

from finance_app.core.services.ingestion import update_consolidated

LOGGER = logging.getLogger(__name__)


def health(_request: HttpRequest) -> HttpResponse:
    return HttpResponse("ok", content_type="text/plain")


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
