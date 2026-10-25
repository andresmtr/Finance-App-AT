from django.urls import path
from . import views

app_name = "ai_chat"

urlpatterns = [
    path("", views.chat_view, name="chat"),
    path("ask/", views.ask_ai, name="ask_ai"),
]
