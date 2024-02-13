from django.urls import path
from . import views

urlpatterns = [
    path("",views.face_re),
    path('receive_image/', views.receive_image, name='recognize-face'),
    path("p_img/",views.process_video,name="p_image")
]