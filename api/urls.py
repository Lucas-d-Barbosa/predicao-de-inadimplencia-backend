from django.urls import path
from . import views

urlpatterns = [
    # Quando alguém acessar 'api/predict/', chame a função 'predict_inadimplencia'
    path('predict/', views.predict_inadimplencia, name='predict'),
]