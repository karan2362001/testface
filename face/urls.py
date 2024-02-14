from django.urls import path
from . import views

urlpatterns = [
    path("",views.run, name='video_feed'),
   # path('receive_image/', views.receive_image, name='recognize-face'),
    #path("p_img/",views.process_video,name="p_image"),
   # path('stop_video_feed/', views.stop_video_feed, name='stop_video_feed'),
    #path('redirected_page/', views.redirected_page, name='redirected_page'),
   path('video_feed/', views.video_feed, name='redirected_page'),

]