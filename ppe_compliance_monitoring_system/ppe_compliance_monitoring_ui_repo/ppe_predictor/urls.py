from django.urls import include, path

from ppe_predictor.views import HomePageView, PredictView

urlpatterns = [
    path('about/', HomePageView.as_view(), name='about'),
path('predict/', PredictView.as_view(), name='about')
]