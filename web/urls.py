from django.conf.urls import url
from . import views

app_name = "predict_app" #강의에는 존재

urlpatterns = [
    #url('main', views.main, name='main'),
    #url('', views.main, name='main'),
    #url('predict', views.predict, name='predict'),
    url('',views.predict, name="predict_page"),
    url('',views.predict_app, name="predict_page2")
    #path('예측/',views.붓꽃예측, name="붓꽃예측"),
]
