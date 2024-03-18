# 颜色空间转换：cvtColor()
# 图像缩放和裁剪：resize()、crop()等
# 图像平滑：blur()、GaussianBlur()等
# 图像形态学操作：erode()、dilate()、morphologyEx()等
# 图像边缘检测：Canny()、Sobel()等
# 直方图均衡化：equalizeHist()
# 图像阈值分割：threshold()、adaptiveThreshold()等
# 图像特征检测和匹配：SIFT、SURF、ORB、BFMatcher、FlannBasedMatcher等
# 目标检测和跟踪：Haar特征分类器、HOG特征+SVM分类器、MeanShift、CamShift等
# 图像分割：分水岭算法、GrabCut算法等
# 图像处理和滤波：filter2D()、getStructuringElement()、sepFilter2D()等
# 视频处理和跟踪：VideoCapture、VideoWriter、cv2.Tracker等

import base64
import io
import os

import cv2
from django.http import JsonResponse, HttpResponse
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pytesseract
from ultralytics import YOLO
import cv2
import json
import base64
import asyncio
import websockets
import pyrtmp
from PIL import Image


def show(img):
    plt.imshow(img[:, :, ::-1])
    plt.show()


def threshold(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        threshold_value = int(request.POST.get('threshold_value', 128))
        max_value = int(request.POST.get('max_value', 255))
        threshold_type = int(request.POST.get('threshold_type', cv2.THRESH_BINARY))

        # 读取图像数据
        # print(image)
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # show(img)

        # 对图像进行处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(img, threshold_value, max_value, threshold_type)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpeg', thresh)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def adaptive_threshold(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        max_value = int(request.POST.get('max_value', 255))
        adaptive_method = int(request.POST.get('adaptive_method', cv2.ADAPTIVE_THRESH_MEAN_C))
        threshold_type = int(request.POST.get('threshold_type', cv2.THRESH_BINARY))
        block_size = int(request.POST.get('block_size', 11))
        C = int(request.POST.get('C', 2))

        # 读取图像数据
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(img, max_value, adaptive_method, threshold_type, block_size, C)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', thresh)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 直方图均衡化
def calcHist(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 灰度图
def grayscale(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 边缘检测
def canny(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        threshold1 = int(request.POST.get('threshold1', 100))
        threshold2 = int(request.POST.get('threshold2', 200))
        edges = request.POST.get('edges', None)
        apertureSize = int(request.POST.get('apertureSize', 3))
        L2gradient = request.POST.get('L2gradient', False) == 'true'
        sobel_x = request.POST.get('sobel_x', False)
        sobel_y = request.POST.get('sobel_y', False)
        ksize = int(request.POST.get('ksize', 3))
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # if edges == 'canny':
        img_edges = cv2.Canny(img_gray, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)
        # elif edges == 'sobel':
        #     if sobel_x:
        #         img_edges = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=ksize)
        #     elif sobel_y:
        #         img_edges = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=ksize)
        #     else:
        #         img_edges = cv2.Sobel(img_gray, cv2.CV_8U, 1, 1, ksize=ksize)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img_edges)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 图像旋转
def rotate(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        angle = int(request.POST['angle'])
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        # 将处理后的图像转换为多种格式的字节
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 颜色转换
def cvt_color(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        code = int(request.POST.get('code', cv2.COLOR_BGR2GRAY))
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        img = cv2.cvtColor(img, code)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 裁剪
def crop(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        x = int(request.POST.get('x', 0))
        y = int(request.POST.get('y', 0))
        width = int(request.POST.get('width', 100))
        height = int(request.POST.get('height', 100))

        # 读取图像
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行裁剪
        crop_img = img[y:y + height, x:x + width]

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', crop_img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 图像平滑
def smooth(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        kernel_size = int(request.POST.get('kernel_size', 3))
        method = request.POST.get('method', 'blur')
        sigmaX = int(request.POST.get('sigmaX', 0))
        sigmaY = int(request.POST.get('sigmaY', 0))
        borderType = int(request.POST.get('borderType', cv2.BORDER_DEFAULT))
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        if method == 'blur':  # 均值滤波
            img = cv2.blur(img, (kernel_size, kernel_size), borderType)
        elif method == 'GaussianBlur':
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX, sigmaY, borderType)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 图像形态学操作
def morphology(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        morphology_type = request.POST.get('morphology_type', 'erode')
        kernel_size = int(request.POST.get('kernel_size', 3))
        kernel_shape = request.POST.get('kernel_shape', 'ellipse')
        iterations = int(request.POST.get('iterations', 1))
        border_type = request.POST.get('border_type', 'default')
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 创建结构元素
        if kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == 'rect':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # 执行形态学操作
        if morphology_type == 'erode':
            img = cv2.erode(img, kernel, iterations=iterations, borderType=get_border_type(border_type))
        elif morphology_type == 'dilate':
            img = cv2.dilate(img, kernel, iterations=iterations, borderType=get_border_type(border_type))
        elif morphology_type == 'open':
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations,
                                   borderType=get_border_type(border_type))
        elif morphology_type == 'close':
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations,
                                   borderType=get_border_type(border_type))
        elif morphology_type == 'gradient':
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iterations,
                                   borderType=get_border_type(border_type))
        else:
            return JsonResponse({'error': 'Invalid morphology type'})

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def get_border_type(border_type):
    if border_type == 'constant':
        return cv2.BORDER_CONSTANT
    elif border_type == 'replicate':
        return cv2.BORDER_REPLICATE
    elif border_type == 'reflect':
        return cv2.BORDER_REFLECT
    elif border_type == 'reflect101':
        return cv2.BORDER_REFLECT_101
    elif border_type == 'wrap':
        return cv2.BORDER_WRAP
    else:
        return cv2.BORDER_DEFAULT


# 图像特征检测和匹配
def sift_match(request):
    if request.method == 'POST':
        # 处理接收的值
        img1 = request.FILES['img1']
        img2 = request.FILES['img2']
        img1_data = img1.read()
        img1_array = np.fromstring(img1_data, np.uint8)
        img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
        img2_data = img2.read()
        img2_array = np.fromstring(img2_data, np.uint8)
        img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)

        # 转换为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 创建SIFT对象，检测关键点和描述符
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # 暴力匹配器(Brute Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

        # 比率测试(Ratio Test)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # 绘制匹配结果
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2, matchColor=None, singlePointColor=None,
                                  matchesMask=None,
                                  drawImg=None, matchThickness=2, lineType=cv2.LINE_AA)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img3)
        response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')
        img1.close()
        img2.close()
        return response
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 图像分割
def image_segmentation(request):
    if request.method == 'POST':
        # 获取上传的图像文件和相关参数
        image = request.FILES.get('file')
        k = int(request.POST.get('k', 3))
        iterations = int(request.POST.get('iterations', 5))
        gamma = float(request.POST.get('gamma', 50))
        gamma_c = float(request.POST.get('gamma_c', 50))

        # 读取图像文件
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 图像分割
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)
        sure_bg = cv2.dilate(opening, kernel, iterations=iterations)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [0, 0, 255]

        line_thickness = 3
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if markers[i, j] == -1:
                    cv2.line(img, (j - line_thickness, i), (j + line_thickness, i), (0, 0, 255),
                             thickness=line_thickness)
                    cv2.line(img, (j, i - line_thickness), (j, i + line_thickness), (0, 0, 255),
                             thickness=line_thickness)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 图像处理和滤波
def filter2d(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        kernel_size = int(request.POST.get('kernel_size', 3))
        kernel_values = request.POST.getlist('kernel_values[]', [])
        border_type = request.POST.get('border_type', 'default')
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        kernel = np.array(kernel_values, np.float32).reshape(kernel_size, kernel_size)
        kernel = kernel / np.sum(kernel)  # 归一化
        img = cv2.filter2D(img, -1, kernel, borderType=border_type)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def get_structuring_element(request):
    if request.method == 'POST':
        # 处理接收的值
        shape = int(request.POST.get('shape', 0))
        ksize = int(request.POST.get('ksize', 3))

        # 获取结构元素
        kernel = cv2.getStructuringElement(shape, (ksize, ksize))

        # 将处理后的结构元素转换为多种格式的字节流
        buffer = cv2.imencode('.txt', kernel)[1]
        response = HttpResponse(buffer.tobytes(), content_type='text/plain')
        return response
    else:
        return JsonResponse({'error': 'Invalid request method'})


def sep_filter2d(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        kernel_size = int(request.POST.get('kernel_size', 3))
        kernel_x_values = request.POST.getlist('kernel_x_values[]', [])
        kernel_y_values = request.POST.getlist('kernel_y_values[]', [])
        border_type = request.POST.get('border_type', 'default')
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        kernel_x = np.array(kernel_x_values, np.float32).reshape(kernel_size, 1)
        kernel_y = np.array(kernel_y_values, np.float32).reshape(1, kernel_size)
        kernel_x = kernel_x / np.sum(kernel_x)  # 归一化
        kernel_y = kernel_y / np.sum(kernel_y)  # 归一化
        img = cv2.sepFilter2D(img, -1, kernel_x, kernel_y, borderType=border_type)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 视频处理和跟踪
def video_tracking(request):
    if request.method == 'POST':
        # 获取请求参数
        video_file = request.FILES['video']
        tracker_type = request.POST.get('tracker_type', 'BOOSTING')
        bbox = tuple(map(int, request.POST.get('bbox', '0,0,0,0').split(',')))

        # 打开视频文件
        video_data = video_file.read()
        video_array = np.frombuffer(video_data, dtype=np.uint8)
        video = cv2.imdecode(video_array, cv2.IMREAD_COLOR)
        tracker = cv2.Tracker_create(tracker_type)

        # 设置追踪对象
        tracker.init(video, bbox)

        # 读取视频并进行跟踪
        while True:
            ret, frame = video.read()
            if not ret:
                break
            success, bbox = tracker.update(frame)

            # 绘制追踪框
            if success:
                x, y, w, h = tuple(map(int, bbox))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # 将处理后的视频帧转换为字节流
            retval, buffer = cv2.imencode('.jpg', frame)
            response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')
            yield response

        video.release()
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 目标检测
def object_detection(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')

        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 对图像进行处理
        # Load class labels
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, 'coco.names'), 'r') as f:
            classNames = [line.strip() for line in f.readlines()]

        # Load pre-trained model
        model = cv2.dnn_DetectionModel(
            '/Users/heboxuan/codeProjects/PycharmProjects/imageDemo/templates/frozen_inference_graph.pb',
            '/Users/heboxuan/codeProjects/PycharmProjects/imageDemo/templates/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

        # Set input image size and scaling factor
        model.setInputSize(320, 320)
        model.setInputScale(1.0 / 127.5)
        model.setInputMean((127.5, 127.5, 127.5))
        model.setInputSwapRB(True)

        # Run object detection
        classIds, confs, bbox = model.detect(image, confThreshold=0.5)

        # Draw bounding boxes and labels on image
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(image, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpeg', image)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 车牌识别
def license_plate_recognition(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')

        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (600, 400))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        edged = cv2.Canny(gray, 30, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
            print("No contour detected")
        else:
            detected = 1

        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        print("programming_fever's License Plate Recognition\n")
        print("Detected license plate Number is:", text)
        img = cv2.resize(img, (500, 300))
        Cropped = cv2.resize(Cropped, (400, 200))

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpeg', Cropped)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        return JsonResponse({
            'image_base64': image_base64,
            'che': text
        })
    else:
        return JsonResponse({'error': 'Invalid request method'})


# def detect(request):
#     if request.method == 'POST':
#         # 获取前端发送的视频流数据
#         image_data = request['image']
#
#         # 将 base64 编码的图像数据解码成图像
#         image_base64 = base64.b64decode(image_data)
#         image_np = np.frombuffer(image_base64, dtype=np.uint8)
#         image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#
#         # 进行目标检测和识别
#         model = YOLO('yolov8n.pt')
#
#         results = model(image, device="mps")
#         result = results[0]
#         box = np.array(result.boxes.xyxy.cpu(), dtype="int")
#         cls = np.array(result.boxes.cls.cpu(), dtype="int")
#         conf = np.array(result.boxes.conf.cpu(), dtype="float")
#
#         # 返回检测结果
#         results = [{'label': cls, 'position': [box[0], box[1], box[2], box[3]], 'confidence': conf}]
#         return JsonResponse({'results': results})


# async def detect(websocket):
#     # cap = cv2.VideoCapture('rtmp://localhost:1935/mytv/room')
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # 对视频流进行处理，获得处理结果
#         result = {'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]]}
#         # 将处理结果转换为JSON格式，并发送给前端
#         await websocket.send(json.dumps(result))
#         asyncio.get_event_loop().run_until_complete(
#             websockets.serve(detect, 'localhost', 1935))


# def rtmp_stream(request):
#     if request.method == 'POST':
#         video_file = request.FILES.get('video')
#         rtmp_url = 'rtmp://localhost:1935/mytv/room'  # 视频流地址
#         rtmp_client = pyrtmp.RtmpClient(rtmp_url, live=True)
#         rtmp_client.connect()
#         rtmp_client.publish('live', 'stream_ƒƒ')
#         rtmp_client.write(video_file.read())
#         rtmp_client.close()
#         return JsonResponse({'status': 'success'})
#     else:
#         return JsonResponse({'status': '失败'})


def detect(request):
    if request.method == 'POST':
        # 从POST请求中解析图像数据
        # 处理接收的值
        image = request.FILES.get('image')
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # plt.imshow(img)
        # plt.show()

        model = YOLO('yolov8n.pt')

        result = model(img, device="mps")[0]
        box = np.array(result.boxes.xywh.cpu(), dtype="int")
        cls = np.array(result.boxes.cls.cpu(), dtype="int")
        conf = np.array(result.boxes.conf.cpu(), dtype="float")
        data = []

        # 遍历每个目标框
        for box, cls, conf in zip(box, cls, conf):
            # 获取目标框左上角和右下角的坐标
            x, y, w, h = box
            label = result.names.get(cls)
            # 获取目标框对应的类别标签和置信度得分
            confidence = float(conf)
            # 将目标框信息添加到列表中
            data.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "label": label,
                "confidence": confidence
            })

        print('----------')
        json_data = json.dumps(data)
        print(json_data)

        return JsonResponse(json_data, safe=False)

    json_data = [
        {
            "x": 100,
            "y": 150,
            "w": 50,
            "h": 80,
            "label": "person",
            "confidence": 0.85
        },
        {
            "x": 250,
            "y": 100,
            "w": 80,
            "h": 60,
            "label": "car",
            "confidence": 0.92
        }
    ]

    return JsonResponse("", safe=False)


# 轮廓检测
def line(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        threshold_value = int(request.POST.get('threshold_value', 128))
        max_value = int(request.POST.get('max_value', 255))
        threshold_type = int(request.POST.get('threshold_type', cv2.THRESH_BINARY))
        model = int(request.POST.get('model', cv2.RETR_EXTERNAL))
        method = int(request.POST.get('method', cv2.CHAIN_APPROX_SIMPLE))

        # 读取图像数据
        # print(image)
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # show(img)

        # 对图像进行处理
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 进行阈值处理
        ret, thresh = cv2.threshold(gray, threshold_value, max_value, threshold_type)

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(thresh, model, method)

        # 创建一个空白图像作为结果
        result = np.zeros_like(img)

        # 绘制轮廓
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        # 将原始图像和轮廓图像合并
        merged = cv2.addWeighted(img, 1, result, 0.5, 0)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpeg', merged)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 直线
def zline(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')

        # 读取图像数据
        # print(image)
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        rho = int(request.POST.get('rho', 1))
        theta = float(request.POST.get('theta', np.pi / 180))
        threshold = int(request.POST.get('threshold', 100))
        threshold1 = int(request.POST.get('threshold1', 100))
        threshold2 = int(request.POST.get('threshold2', 200))
        edges = request.POST.get('edges', None)
        apertureSize = int(request.POST.get('apertureSize', 3))
        L2gradient = request.POST.get('L2gradient', False) == 'true'
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 进行边缘检测
        edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

        # 进行直线检测
        lines = cv2.HoughLines(edges, rho, theta, threshold=threshold)

        # 创建一个空白图像作为结果
        result = np.zeros_like(img)

        # 绘制检测到的直线
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 将原始图像和直线检测结果合并
        merged = cv2.addWeighted(img, 1, result, 0.5, 0)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpeg', merged)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})


# 角点
def points(request):
    if request.method == 'POST':
        # 处理接收的值
        image = request.FILES.get('file')
        blockSize = int(request.POST.get('blockSize', 2))
        ksize = int(request.POST.get('ksize', 3))
        k = float(request.POST.get('k', 0.04))

        # 读取图像数据
        # print(image)
        image_data = image.read()
        image_array = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 执行Harris角点检测
        dst = cv2.cornerHarris(gray, blockSize, ksize, k)

        # 标记角点
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 将原始图像和角点检测结果合并
        # merged = np.concatenate((image, img), axis=1)

        # 将处理后的图像转换为多种格式的字节流
        retval, buffer = cv2.imencode('.jpeg', img)
        image_data = buffer.tobytes()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        image.close()
        return JsonResponse({'image_base64': image_base64})
    else:
        return JsonResponse({'error': 'Invalid request method'})
