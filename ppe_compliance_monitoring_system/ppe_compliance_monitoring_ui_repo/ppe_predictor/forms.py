from django import forms


class PredictForm(forms.Form):
    file = forms.FileField(label='Please upload an image',widget=forms.ClearableFileInput(attrs={'multiple': False,'class':'form-control'}))

