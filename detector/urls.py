from django.urls import path
from . import views, multimodal_views

urlpatterns = [
    path('', views.home, name='home'),
    path('multimodal/', multimodal_views.home, name='multimodal_home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predict/url/', views.predict_url, name='predict_url'),
    path('predict/sms/', views.predict_sms, name='predict_sms'),
    path('predict/email/', views.predict_email, name='predict_email'),
    # Multi-modal endpoints
    path('predict/multimodal/', multimodal_views.predict_multimodal, name='predict_multimodal'),
    path('upload-analyze/', multimodal_views.upload_and_analyze, name='upload_analyze'),
    path('predict/batch/', multimodal_views.batch_analyze, name='batch_analyze'),
    path('capabilities/', multimodal_views.capabilities, name='capabilities'),
    # Existing endpoints
    path('clear-history/', views.clear_history, name='clear_history'),
    path('delete-prediction/<uuid:prediction_id>/', views.delete_prediction, name='delete_prediction'),
    path('delete-selected-history/', views.delete_selected_history, name='delete_selected_history'),
] 