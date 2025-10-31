"""
URL patterns for pattern examples
"""
from django.urls import path
from . import views

app_name = 'patterns'

urlpatterns = [
    path('', views.index, name='index'),
    path('dependent-dropdowns/', views.dependent_dropdowns, name='dependent_dropdowns'),
    path('conditional-visibility/', views.conditional_visibility, name='conditional_visibility'),
    path('state-restoration/', views.state_restoration, name='state_restoration'),

    # HTMX endpoints
    path('api/get-dataset-columns/', views.get_dataset_columns, name='get_dataset_columns'),
    path('api/get-test-parameters/', views.get_test_parameters, name='get_test_parameters'),
    path('api/get-distribution-params/', views.get_distribution_params, name='get_distribution_params'),
]
