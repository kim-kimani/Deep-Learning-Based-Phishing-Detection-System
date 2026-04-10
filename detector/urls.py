from django.urls import path
from . import views, multimodal_views, enhanced_views

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
    # Enhanced dashboard endpoints
    path('enhanced-dashboard/', enhanced_views.enhanced_predictions_dashboard, name='enhanced_predictions_dashboard'),
    path('prediction/<uuid:prediction_id>/', enhanced_views.prediction_detail, name='prediction_detail'),
    path('prediction/<uuid:prediction_id>/manual/', enhanced_views.manual_analysis_page, name='manual_analysis_page'),
    path('prediction/<uuid:prediction_id>/deepseek/', enhanced_views.deepseek_analysis_page, name='deepseek_analysis_page'),
    path('api/prediction-stats/', enhanced_views.get_prediction_stats, name='get_prediction_stats'),
    path('api/delete-prediction/<uuid:prediction_id>/', enhanced_views.delete_prediction_api, name='delete_prediction_api'),
    path('api/bulk-delete-predictions/', enhanced_views.bulk_delete_predictions, name='bulk_delete_predictions'),
    # Email-focused dashboard and action endpoints
    path('email-dashboard/', enhanced_views.email_dashboard, name='email_dashboard'),
    path('fetch-emails/', enhanced_views.fetch_emails_view, name='fetch_emails'),
    path('analyze-emails-deepseek/', enhanced_views.analyze_emails_deepseek_view, name='analyze_emails_deepseek'),
    path('api/process-status/<str:process_id>/', enhanced_views.get_process_status, name='get_process_status'),
    path('api/reanalyze-email/<uuid:prediction_id>/', enhanced_views.reanalyze_email_api, name='reanalyze_email_api'),
    path('report-file/', enhanced_views.serve_report_file, name='serve_report_file'),
    path('prediction/<str:prediction_id>/pdf/', enhanced_views.report_pdf, name='report_pdf'),
]