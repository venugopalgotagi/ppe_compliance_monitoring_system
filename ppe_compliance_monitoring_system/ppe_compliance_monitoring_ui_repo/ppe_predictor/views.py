import requests
from django.shortcuts import render
from django.views.generic import TemplateView, View

from ppe_predictor.forms import PredictForm


# Create your views here.
import json

class HomePageView(TemplateView):
    template_name = "ppe_predictor/home.html"

    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        context['title'] = 'Home Page'
        return context



class PredictView(View):
    template_name = "ppe_predictor/predict.html"

    def get(self, request):
        form = PredictForm()
        return render(request, self.template_name, context={'form': form})

    def post(self, request):
        predictor_response = None
        form = PredictForm(files = request.FILES)

        if form.is_valid():
            predictor_response = requests.post(url="http://127.0.0.1:8080/predict_img", files={'img':form.files.get('file').file})
            predictor_response.raise_for_status()
        else:
            print("form not valid")
            form = PredictForm()
        print("predictor response:", json.dumps(predictor_response.json(), indent=4))
        return render(request, self.template_name, context={'form': form, 'predictor_response': predictor_response.json()})


