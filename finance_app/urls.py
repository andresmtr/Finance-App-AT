from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include
from django.urls import path

from finance_app.core import views as core_views
from finance_app.dashboard import views as dashboard_views
from finance_app.transactions import views as transaction_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("i18n/", include("django.conf.urls.i18n")),
    path("login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("signup/", core_views.signup, name="signup"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("", dashboard_views.dashboard, name="dashboard"),
    path("ai-chat/", include("finance_app.ai_chat.urls", namespace="ai_chat")),
    path("transactions/", transaction_views.transactions, name="transactions"),
    path("imports/", transaction_views.import_pdfs, name="import_pdfs"),
    path("imports/<int:batch_id>/review/", transaction_views.review_next, name="review_next"),
    path("transactions/manual/", transaction_views.manual_transaction, name="manual_transaction"),
    path("transactions/<int:pk>/edit/", transaction_views.edit_transaction, name="edit_transaction"),
    path("transactions/<int:pk>/source/", transaction_views.transaction_source, name="transaction_source"),
    path("transactions/<int:pk>/delete/", transaction_views.delete_transaction, name="delete_transaction"),
    path("telegram/webhook/", transaction_views.telegram_webhook, name="telegram_webhook"),
    path("update/", core_views.update_data, name="update_data"),
    path("backup/download/", core_views.download_backup, name="download_backup"),
    path("backup/restore/", core_views.restore_backup, name="restore_backup"),
    path("health/", core_views.health, name="health"),
]

# Financial source documents are served only through the authenticated source view.
