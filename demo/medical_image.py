# import imghdr
# import cv2
# from django.http import HttpResponseBadRequest
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# import pydicom
# from PIL import Image
#
#
# class MedicalImageMiddleware:
#     def __init__(self, get_response):
#         self.get_response = get_response
#
#     def __call__(self, request):
#         # 获取请求中的文件
#         if request.method == 'POST':
#             file = request.FILES.get('file')
#             if file:
#                 # 检查文件类型是否为医学图像格式
#                 if imghdr.what(file) in ['dicom']:
#                     # 读取DICOM文件
#                     dicom = pydicom.dcmread(file.path)
#
#                     # 获取像素数组和图像大小
#                     pixel_array = dicom.pixel_array
#                     width, height = pixel_array.shape
#
#                     # 将像素数组转换为图像
#                     image = Image.fromarray(pixel_array)
#                     # 在这里进行OpenCV处理
#                     # ...
#                     # 将处理后的图像写入新的文件并替换原始文件
#                     processed_path = default_storage.save('file',
#                                                           ContentFile(cv2.imencode('.png', image)[1]))
#                     file = default_storage.open(processed_path)
#         response = self.get_response(request)
#         return response
