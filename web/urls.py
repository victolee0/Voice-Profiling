from django.conf.urls import url
from . import views

app_name = "predict_app" #강의에는 존재

urlpatterns = [
    url('',views.predict, name="predict_page"),
    url('',views.predict_app, name="predict_page2")
]
