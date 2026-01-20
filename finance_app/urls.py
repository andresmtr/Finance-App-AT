from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path

from finance_app.core import views as core_views
from finance_app.dashboard import views as dashboard_views
from finance_app.transactions import views as transaction_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("login/", auth_views.LoginView.as_view(template_name="registration/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("", dashboard_views.dashboard, name="dashboard"),
    path("transactions/", transaction_views.transactions, name="transactions"),
    path("update/", core_views.update_data, name="update_data"),
    path("health/", core_views.health, name="health"),
]
