import json
import logging
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .services.ai_provider import AIProvider, AIProviderError
from .services.finance_query import resolve_finance_question

logger = logging.getLogger(__name__)


def _chat_model_options():
    options = [{"value": "local", "label": "Consulta local"}]
    openai_model = getattr(settings, "OPENAI_CHAT_MODEL", "").strip()
    gemini_model = getattr(settings, "GEMINI_CHAT_MODEL", "").strip()
    if openai_model:
        options.append({"value": f"openai:{openai_model}", "label": f"OpenAI - {openai_model}"})
    if gemini_model:
        options.append({"value": f"gemini:{gemini_model}", "label": f"Gemini - {gemini_model}"})
    return options


@login_required
def chat_view(request):
    return render(request, "ai_chat/chat.html", {"chat_model_options": _chat_model_options()})


def _json_error(message, status=400):
    return JsonResponse({"error": message}, status=status)


def _parse_model_choice(value):
    value = (value or "local").strip()
    allowed_values = {option["value"] for option in _chat_model_options()}
    if value not in allowed_values:
        return "local", "", "local"
    if value == "local":
        return "local", "", "local"
    provider, model = value.split(":", 1)
    return provider, model, value


def _answer_with_model(question, result, provider, model):
    fallback = result["answer"]
    if provider == "local" or not result.get("valid"):
        return fallback, "local", ""

    try:
        ai = AIProvider(provider=provider)
        ai.model = model
        system_prompt = (
            "Eres un asistente financiero personal. "
            "Responde en espanol con maximo 3 frases cortas. "
            "Usa solamente los datos del JSON entregado; no inventes cifras, fechas ni transacciones. "
            "Si los datos no tienen filas, dilo claramente."
        )
        user_prompt = (
            f"Pregunta del usuario: {question}\n"
            f"Intencion detectada: {result['intent']}\n"
            f"Respuesta local base: {fallback}\n"
            "Datos calculados desde la base de datos:\n"
            f"{json.dumps(result['payload'], ensure_ascii=False)}"
        )
        answer = (ai.get_response(system_prompt, user_prompt) or "").strip()
        if answer:
            return answer[:900], f"{provider}:{model}", ""
    except AIProviderError as exc:
        return fallback, "local", str(exc)
    except Exception as exc:
        logger.warning("Selected chat model failed: %s", exc)
        return fallback, "local", f"No fue posible usar {provider}:{model}. Respondi con consulta local."
    return fallback, "local", "El modelo no devolvio respuesta. Respondi con consulta local."


@login_required
def ask_ai(request):
    if request.method != "POST":
        return _json_error("Metodo no permitido.", status=405)

    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return _json_error("JSON invalido.")

    user_query = (data.get("query") or "").strip()
    provider, model, requested_model = _parse_model_choice(data.get("model"))

    if not user_query:
        return _json_error("Escribe una pregunta para el asistente.")

    try:
        result = resolve_finance_question(user_query, user_id=request.user.id)
        summary, model_used, model_warning = _answer_with_model(user_query, result, provider, model)
        return JsonResponse({
            "summary": summary,
            "intent": result["intent"],
            "valid": result["valid"],
            "payload": result["payload"],
            "row_count": result["payload"].get("row_count", 0),
            "requested_model": requested_model,
            "model_used": model_used,
            "model_warning": model_warning,
        })
    except Exception as exc:
        logger.exception("AI chat request failed")
        return _json_error(f"No pude completar la consulta: {exc}", status=500)
