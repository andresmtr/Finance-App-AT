from django.conf import settings
from django.conf.urls.static import static
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
    path("transactions/", transaction_views.transactions, name="transactions"),
    path("imports/", transaction_views.import_pdfs, name="import_pdfs"),
    path("imports/<int:batch_id>/review/", transaction_views.review_next, name="review_next"),
    path("transactions/manual/", transaction_views.manual_transaction, name="manual_transaction"),
    path("transactions/<int:pk>/edit/", transaction_views.edit_transaction, name="edit_transaction"),
    path("transactions/<int:pk>/delete/", transaction_views.delete_transaction, name="delete_transaction"),
    path("update/", core_views.update_data, name="update_data"),
    path("health/", core_views.health, name="health"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
