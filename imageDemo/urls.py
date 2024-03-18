"""imageDemo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from demo.views import *
from rest_framework.documentation import include_docs_urls
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# 使用 DRF YASG 自动生成 API 文档
schema_view = get_schema_view(
    openapi.Info(
        title="API文档",
        default_version='v1',
        description="这是API文档",
        terms_of_service="https://www.example.com/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)


urlpatterns = [
    path('swagger(<format>\.json|\.yaml)', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

    # path('admin/', admin.site.urls),
    path('threshold/', threshold),
    path('adaptive_threshold/', adaptive_threshold),
    path('calcHist/', calcHist),  # 直方图均衡化
    path('grayscale/', grayscale),  # 灰度图
    path('canny/', canny),  # 边缘检测
    path('rotate/', rotate),  # 图像旋转
    path('cvt_color/', cvt_color),  # 颜色转换
    path('crop/', crop),  # 图像裁剪
    path('smooth/', smooth),  # 图像平滑
    path('morphology/', morphology),  # 图像形态学操作
    path('sift_match/', sift_match),  # 图像特征检测和匹配
    path('image_segmentation/', image_segmentation),  # 图像分割
    path('filter2d/', filter2d),  # 图像处理和滤波
    path('video_tracking/', video_tracking),  # 视频跟踪
    path('object_detection/', object_detection),  # 目标检测
    path('license_plate_recognition/', license_plate_recognition),  # 车牌识别
    path('detect/', detect),
    path('line/', line),
    path('zline/', zline),
    path('points/', points),

]

