"""
URL configuration for htmx-alpine-patterns project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('patterns/', include('app.urls')),
]
