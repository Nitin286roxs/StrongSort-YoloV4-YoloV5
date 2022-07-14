#ROI For tyre counting
#polygon_roi = [[405,220], [1075,237], [1220,915], [275,920]]
line = [(630,360), (600,950)]
#ROI for Bilboard
#ROI for 1280, 720
#polygon_roi = [[435,130], [775,125], [1020,720], [210,720]]
#ROI for 1920 , 1080 
polygon_roi = [[750,425], [1295,425], [1475,1290], [500,1290]]
#polygon_roi = [[770,330],[1705,330],[1705, 1290], [770, 1290]]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

import argparse
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
try:
    RUN_MODE = int(os.environ["RUN_MODE"])
except KeyError:
    RUN_MODE = 0

import sys
import base64
import threading
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import json
import math
import random

from numpy import linalg as LA
import cv2
from scipy.spatial import distance
from openvino.inference_engine import IECore

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH

#setting path for gaze estimation
if str(ROOT / 'gaze_estimation') not in sys.path:
    sys.path.append(str(ROOT / 'gaze_estimation'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yaml_reader import YamlReader
from yolov4.models.models import *
from yolov4.utils.datasets import *
from yolov4.utils.datasets import LoadStreamsV4, LoadImagesV4
from yolov4.utils.general import *
from yolov4.utils.general import (
    check_img_sizeV4, non_max_suppressionV4, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from gaze_estimation import get_eye_box
#TODO Adding api Layer
from extended_apilayer import ExtApiLayer, add_method

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

import socket
import imagezmq
LOCK = threading.Lock()
sender = None
if RUN_MODE == 0:
    sender = imagezmq.ImageSender(connect_to='tcp://*:5555', REQ_REP=False)
    host_name = 'From Sender' #socket.gethostname()

genderNet = None
eyes_cascade = None
USECASE = (os.environ["USECASE"])
if USECASE=="Billboard":
    #Gender model
    genderProto = "gender_model/gender_deploy.prototxt"
    genderModel = "gender_model/gender_net.caffemodel"
    print(f"{genderProto} exists :{os.path.isfile(genderProto)}")
    print(f"{genderModel} exists :{os.path.isfile(genderModel)}")
    #Eye Cascade Model
    eyes_cascade_name = "eye_detection_model/haarcascade_eye_tree_eyeglasses.xml"
    print(f"{eyes_cascade_name} exists :{os.path.isfile(eyes_cascade_name)}")
    eyes_cascade = cv2.CascadeClassifier()
    #-- 1. Load the cascades
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    #Lodd network
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    print(f"genderNet: {genderNet}")


@add_method(ExtApiLayer)
def process_input(md, img_string_tuples):
    global MD, IMG_STRING_TUPLES, RESPONSE, RESPONSE_SENT
    LOCK.acquire()
    print("Lock Acquired !!")
    print(f"md: {md}")
    print(f"length of image string: {img_string_tuples[0]['len']}")
    MD = md
    IMG_STRING_TUPLES = img_string_tuples
    RESPONSE = None
    RESPONSE_SENT = False
    while not RESPONSE:
        time.sleep(.001)
    RESPONSE_SENT = True
    #TO PREVENT OVERRIDE RESPONSE
    COPY_RESPONSE = RESPONSE
    LOCK.release()
    return COPY_RESPONSE 
def automate_histogram_mid(mean):
    # For Darker Image
    if mean < 75:
        mid = random.choice([0.5, 0.6, 0.7])
    # For Average Lighting Image
    elif mean >= 75 and mean <= 130:
        mid = random.choice([.2,.3,.4, 0.5, 0.6, 0.7])
    # For Brighter Image
    else:
        mid = random.choice([0.2, 0.3, 0.4,.5])
    return mid

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    # apply gamma correction using the lookup table
    return cv2.LUT(image, lookUpTable)

#Intersection over union check for two objects
def iou_check(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return interArea / float(boxAArea) #+ boxBArea - interArea)

def letterbox_V5(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def letterbox_V4(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

unique_tyre_centroid = {} 
def process_frame(im0, stride, nr_sources, USECASE,  model, names, device, half,\
                  model_version, visualize, roi, pt, conf_thres, iou_thres, classes, agnostic_nms,\
                  cfg, strongsort_list, save_vid, save_crop, save_txt, show_vid, max_det, img_size=640):
    #TODO
    '''
    Cropping the ROI part from image dataset for inference to improve tracking
    '''
    global unique_tyre_centroid
    assert im0 is not None, 'Image Not Found ' + path
    background = np.full((im0.shape[0], im0.shape[1], 3) , (114,114,114), np.uint8)
    overlay = im0[roi[1]:roi[3], roi[0]:roi[2]]
    background[roi[1]:roi[3], roi[0]:roi[2]] = overlay
    auto_size=64
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    outputs = [None] * nr_sources
    augment = False
    hide_labels = False
    hide_conf = False
    hide_class = False
    # Padded resize
    if model_version == "yoloV5":
        im = letterbox_V5(background, img_size, stride=stride, auto=pt)[0]
    else:
        im = letterbox_V4(background, new_shape=img_size, auto_size=auto_size)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    #return path, img, img0, self.cap, s
 
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    #visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
    if model_version=="yolov5":
        pred = model(im, augment=augment, visualize=visualize)
    else:
        pred = model(im, augment=augment)
    t3 = time_sync()
    dt[1] += t3 - t2

    # Apply NMS
    if model_version=="yolov5":
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    else:
        pred = non_max_suppressionV4(pred, conf_thres, iou_thres, classes, agnostic_nms)
    dt[2] += time_sync() - t3

    # Process detections
    count = 0
    for i, det in enumerate(pred):  # detections per image
        seen += 1
        print(f"i: {i}")
        curr_frames[i] = im0

        #txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
        #s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        #im0 = cv2.rectangle(im0, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
        pts = np.array(polygon_roi,np.int32)
        isEyeDetection = None
        isGenderClassification = None
        isEyeVisible = None
        if USECASE == "Billboard":
            isEyeDetection = True
            isGenderClassification = True
            isEyeVisible = False
        pts = pts.reshape((-1, 1, 2))
        isClosed = True
        color = (0, 255, 0)
        thickness = 2
        im0 = cv2.polylines(im0, [pts],
                  isClosed, color, thickness)
        im0 = cv2.line(im0, line[0],line[1], (255,0,0), thickness)
        annotator = Annotator(im0, isEyeDetection, isGenderClassification, line_width=2, pil=not ascii)
        if prev_frames[i] is None:
            prev_frames[i] = curr_frames[i].copy()
        if cfg.STRONGSORT.ECC:  # camera motion compensation
            strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            print(f"Detection box before scaling: {det}")
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            print(f"Detection box after scaling: {det}")
            
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            # Write results
            #for *xyxy, conf, cls in det:
            #    xywhs_norma = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #print(f"yolo co-ord : {xywhs}")
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to strongsort
            t4 = time_sync()
            outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4

            # draw boxes for visualization
            if len(outputs[i]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                    print(f"Detection box after tracker: {output}")
                    bboxes = output[0:4]
                    #print(f"yolo deepsort co-ord : {bboxes}")
                    id = output[4]
                    cls = output[5]
                    if isEyeDetection and isGenderClassification:
                        face = im0[int(max(0,bboxes[1])):int(min(bboxes[3],im0.shape[0]-1)),int(max(0,bboxes[0])):int(min(bboxes[2], im0.shape[1]-1))]
                        #TODO 
                        '''
                        Applying gamma correction on face.
                        '''
                        count += 1
                        face_temp_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        mean = np.mean(face_temp_gray)
                        mid = automate_histogram_mid(mean)
                        gamma = math.log(mid)/math.log(mean/255)
                        # do gamma correction
                        face_gamma = adjust_gamma(face, gamma)
                        #print(face.shape)
                        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds = genderNet.forward()
                        gender = genderList[genderPreds[0].argmax()]
                        genderConf = genderPreds[0].max()
                        #Adding grayscaled face patch to original image
                        img0_gamma = im0.copy()
                        img0_gamma[int(max(0,bboxes[1])):int(min(bboxes[3],im0.shape[0]-1)),int(max(0,bboxes[0])):int(min(bboxes[2], im0.shape[1]-1))] = face_gamma
                        frame_gray = cv2.cvtColor(img0_gamma, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.equalizeHist(frame_gray)
                        faceROI = frame_gray[int(max(0,bboxes[1])):int(min(bboxes[3],im0.shape[0]-1)),int(max(0,bboxes[0])):int(min(bboxes[2], im0.shape[1]-1))]
                        #if True:#dump_gamma_corrected:
                        #    cv2.imwrite(f"/WS/gamma_corrected/{count}.jpg", face)
                        #    cv2.imwrite(f"/WS/gamma_corrected/{count}_greyscale.jpg", face_temp_gray)
                        #    cv2.imwrite(f"/WS/gamma_corrected/{count}_gamma.jpg", face_gamma)
                        #    cv2.imwrite(f"/WS/gamma_corrected/{count}_org_full.jpg", im0)
                        #    cv2.imwrite(f"/WS/gamma_corrected/{count}_gamma_full.jpg", img0_gamma)
                        eyes = eyes_cascade.detectMultiScale(faceROI,1.3, 5)
                        #print(f"eyes: {eyes}")
                        if len(eyes):
                            is_eye_visible = True
                        else:
                            is_eye_visible = False
                        for (x2,y2,w2,h2) in eyes:
                            eye_center = (int(bboxes[0] + x2 + w2//2), int(bboxes[1] + y2 + h2//2))
                            radius = int(round((w2 + h2)*0.25))
                            #print(f"eye_center: {eye_center}")
                            #print(f"eyes radius: {radius}")
                            im0 = cv2.circle(im0, eye_center, radius, (255, 255, 0 ), 4)
                    if save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                    if save_vid or save_crop or show_vid:  # Add bbox to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        roi = [polygon_roi[0][0], polygon_roi[0][1],\
                                             polygon_roi[2][0], polygon_roi[2][1]]
                        #iou_conf = iou_check(bboxes, roi)
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        iou_conf = iou_check(bboxes, roi)
                        #print(f"iou_conf: {iou_conf}")
                        if iou_conf > 0.5:
                            if USECASE == "Billboard":
                                annotator.box_label_billboard(bboxes, gender, genderConf, is_eye_visible, label, color=colors(c, True))
                            if USECASE == "TyreCounting":
                                tyre_id = f'{id:0.0f}'
                                if tyre_id not in unique_tyre_centroid:
                                    unique_tyre_centroid = {}
                                    unique_tyre_centroid[tyre_id] = {"trail-path": [], "isLoaded": False}
                                centroid = int(bboxes[0]) , int((bboxes[1] + bboxes[3])// 2)
                                unique_tyre_centroid[tyre_id]["trail-path"].append(centroid)
                                for point in unique_tyre_centroid[tyre_id]["trail-path"]:
                                    print(f"point: {point}")
                                    annotator.im = cv2.circle(annotator.im, point, 5, (255,0,255), thickness=3)
                                #TODO Line Crossing Logic
                                '''
                                If mid poit of tyre's TOP-LEFT and BOTTOM-LEFT cross the drawn line then We can say tyre is 
                                loaded.
                                '''
                                #TODO 
                                '''
                                For Futur, We will implement
                                ENTRY and EXIT Logic 
                                '''
                                '''
                                Equation of line
                                A(X1, Y1), B(X2, Y2)
                                '''
                                if not unique_tyre_centroid[tyre_id]["isLoaded"]:
                                    (X1,Y1), (X2,Y2) = line[0], line[1]
                                    PX,PY = centroid
                                    Z = PX*(Y2-Y1)-PY*(X2-X1)-X1*(Y2-Y1)+Y1*(X2-X1)
                                    if Z <= 0:
                                        #Centroid is on or left side of line, means Tyre is loaded
                                        unique_tyre_centroid[tyre_id]["isLoaded"] = True
                                annotator.box_label_tyrecounting(bboxes, 1.0,unique_tyre_centroid[tyre_id]["isLoaded"], label, color=colors(c, True))
                        if save_crop:
                            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

            LOGGER.info(f'Inference Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

        else:
            strongsort_list[i].increment_ages()
            LOGGER.info('No detections')

    return im0

@torch.no_grad()
def run(objyaml):
    global MD, RESPONSE, RESPONSE_SENT, IMG_STRING_TUPLES
    source= objyaml.source
    yolo_weights = str(objyaml.yolo_weights)  # model.pt path(s),
    strong_sort_weights = objyaml.strong_sort  # model.pt path,
    config_strongsort = 'strong_sort/configs/strong_sort.yaml'
    imgsz = (640, 640) # inference size (height, width)
    conf_thres = objyaml.conf_thres  # confidence threshold
    iou_thres = objyaml.iou_thres  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = objyaml.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    show_vid = False  # show results
    save_txt = False  # save results to *.txt
    save_conf = False  # save confidences in --save-txt labels
    save_crop = False  # save cropped prediction boxes
    save_vid = objyaml.save_vid  # save confidences in --save-txt labels
    nosave = False  # do not save images/videos
    if USECASE=="Billboard":
        classes = 1  # head class
    else:
        classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False,  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    print(f"objyaml.project: {objyaml.project}")
    project = "/app/"+objyaml.project  # save results to project/name
    #project = '/app/test/'  # save results to project/name
    name = 'exp'  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    hide_class = False  # hide IDs
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    model_version = objyaml.model_version #yolo model-type yolov4 or v5
    cfg = objyaml.cfg #Original cfg file if model type is yolov4
    fps = objyaml.fps #Input framesrate
    usecase = objyaml.usecase #POC UseCase
    eye_model_type = objyaml.eye_model_type
    #source = str(objyaml.source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = yolo_weights[0].split(".")[0]
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    half = device.type != 'cpu'
    #device = select_device(device)
    stride = None
    pt = None
    if model_version == "yolov5":
        model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        print(f"model_Stride: {stride}, model_name: {names} and model_pt: {pt}")
        if USECASE == "Billboard":
            if(device.type == 'cpu'):
                genderNet.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
                print('Using CPU device.')
            elif(device.type == 'gpu'):
                genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print('Using GPU device.')
    elif model_version == "yolov4":
        model = Darknet(cfg, imgsz).cuda()
        try:
            print(f"yolo_weights: {yolo_weights[0]}")
            model.load_state_dict(torch.load(yolo_weights, map_location=device)['model'])
            print("model loaded")
            names = "tyre"
            #stride, names, pt = model.stride, model.names, model.pt
            #print(f"model_Stride: {model.stride.max()}") # model_name: {names} and model_pt: {pt}")
            #model = attempt_load(weights, map_location=device)  # load FP32 model
            imgsz = (640,640)#check_img_sizeV4(imgsz, s=model.stride.max())  # check img_size
        except:
            print("I am here!!")
            load_darknet_weights(model, yolo_weights)
        model.to(device).eval()
        if half:
            model.half()  # to FP16
    else:
        names = "head"
    #imgsz = check_img_size(imgsz, s=stride)  # check image size

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    nr_sources = 1
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    
    # Dataloader
    #TODO 
    '''
    Intialilze API Layer
    '''
    NODENAME = "tyre_counting"
    if RUN_MODE:
        apil = ExtApiLayer(name=NODENAME, port=5011, max_hung_time=5,\
            log_level=10) #objYaml.values["debugLevel"])
        apil.run()
        #Running infinite loop for input
        frame_count = 0
        while(True):
            print("inside loop main()")
            if RESPONSE_SENT and RUN_MODE:
                time.sleep(.001)
                continue
            #if RUN_MODE:
            #print("not (MD and IMG_STRING_TUPLES) and RUN_MODE: "+str(not (MD and IMG_STRING_TUPLES) and RUN_MODE))
            print(f"IMG_STRING_TUPLES: {IMG_STRING_TUPLES}")
            print(f"MD: {MD}")
            while not (MD and IMG_STRING_TUPLES) and RUN_MODE:
                #print(f"IMG_STRING_TUPLES: {IMG_STRING_TUPLES}")
                #print(f"MD: {MD}")
                #print("not (MD and IMG_STRING_TUPLES) and RUN_MODE: "+str(not (MD and IMG_STRING_TUPLES) and RUN_MODE))
                #LOGGER.info("Waiting for API request going to sleep for 1 msec.")
                time.sleep(0.001)
            print('Receive Status from API for')
            input_json = MD
            img_list_tuples = IMG_STRING_TUPLES
            start = int(time.time() * 1000)
            #TODO 
            '''
            Decode base64 image ndarray
            '''
            temp_start1 = time.time()*1000
            image_string = img_list_tuples[0]["image_string"]
            print("Length of image "+str(i+1)+": "+str(len(image_string)))
            encode_start = time.time()*1000
            jpg_original = base64.b64decode(image_string.encode("utf-8"))
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img_matlab = cv2.imdecode(jpg_as_np, flags=1)
            encode_end = time.time()*1000
            print(f"time taken to decode img: {encode_end - encode_start}")
            roi = [polygon_roi[0][0], polygon_roi[0][1],\
                                                 polygon_roi[2][0], polygon_roi[2][1]]
            #TODO
            '''
            performing yolo inference and tracking
            '''
            im0 = process_frame(img_matlab, stride, nr_sources, USECASE, model, names, device, half, model_version,visualize,\
                          roi, pt, conf_thres, iou_thres, classes,agnostic_nms, cfg, \
                          strongsort_list, save_vid, save_crop, save_txt, show_vid, max_det=max_det, img_size=640)
            #frame_name = f"/WS/frame_dumps/debug_frame_{frame_count}.jpg"
            #cv2.imwrite(frame_name, im0)
            encode_start = time.time()*1000
            img_str = base64.b64encode(cv2.imencode('.bmp', im0)[1]).decode("utf-8")
            encode_end = time.time()*1000
            print(f"time taken to encode img: {encode_end - encode_start}")
            #print(f"{frame_name} exists: {os.path.isfile(frame_name)}")
            MD["receiver"] = MD['sender']+ " world!!"
            input_json = {"image_string": img_str, "Receiver_data": MD}
            send_json = json.dumps(input_json)
            MD =None
            IMG_STRING_TUPLES = None
            RESPONSE = send_json
            frame_count += 1
    else:
        # Dataloader
        nr_sources = 0
        if webcam:
            print("I'm checking webcam!!")
            #show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            if model_version=="yolov5":
                dataset = LoadStreams(polygon_roi, source ,img_size=imgsz, stride=stride, auto=pt)
                nr_sources = len(dataset)
            else:
                dataset = LoadStreamsV4(polygon_roi, source, img_size=imgsz)
                nr_sources = 1
            #nr_sources = len(dataset)
        else:
            print("I'm Here!!")
            if model_version=="yolov5":
                dataset = LoadImages(polygon_roi, source, img_size=imgsz, stride=stride, auto=pt)
            elif model_version=="yolov4":
                save_img = True
                dataset = LoadImagesV4(polygon_roi, source, img_size=imgsz, auto_size=64)
            elif model_version=="openvino":
                #imgsz = (384,672)
                dataset = LoadImagesV4(polygon_roi, source, img_size=imgsz, auto_size=64)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        outputs = [None] * nr_sources
        unique_tyre_centroid = {}
        unique_tyre = []
        unique_person_seen_bilboard = []
        # Run tracking
        if model_version=="yolov5":
            model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        #else:
        #    im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=device)  # input
        #    model.forward(im)
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
        print(f"nr_sources: {nr_sources}, curr_frames: {curr_frames} and prev_frames: {prev_frames}")
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            print(f"s: {s}")
            print(f"im shape: {(im[0]).shape}")
            print(f"im0 shape: {(im0s[0]).shape}")
            t1 = time_sync()
            if model_version != "openvino": 
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            if model_version=="yolov5":
                pred = model(im, augment=augment, visualize=visualize)
            elif model_version=="yolov4":
                pred = model(im, augment=augment)
            else:
                im0s, pred, is_eye_visible = get_eye_box(None, im0s)
                print(f"openvino face detection: {pred}")
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            if model_version=="yolov5":
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            elif model_version=="yolov4":
                pred = non_max_suppressionV4(pred, conf_thres, iou_thres, classes, agnostic_nms)
            dt[2] += time_sync() - t3
            print(f"yolo detection: {pred}")
            # Process detections
            im0 = im0s.copy()
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                print(f"i: {i}")
                curr_frames[i] = im0

                txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                print(f"im.shape[2:]: {im.shape[2:]}")
                #s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                #im0 = cv2.rectangle(im0, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
                pts = np.array(polygon_roi,np.int32)
                pts = pts.reshape((-1, 1, 2))
                isClosed = True
                color = (0, 255, 0)
                thickness = 2
                im0 = cv2.polylines(im0, [pts],
                          isClosed, color, thickness)
                if USECASE != "Billboard":
                    im0 = cv2.line(im0, line[0],line[1], (255,0,0), thickness)
                isEyeDetection = None
                isGenderClassification = None
                isEyeVisible = None
                if USECASE == "Billboard":
                    isEyeDetection = True
                    isGenderClassification = True
                    isEyeVisible = False
                annotator = Annotator(im0, isEyeDetection, isGenderClassification, line_width=2, pil=not ascii)
                if cfg.STRONGSORT.ECC:  # camera motion compensation
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    print(f"detection box before scaling: {det}")
                    if model_version!="openvino": 
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    print(f"detection box after scaling: {det}")
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    # Write results
                    #for *xyxy, conf, cls in det:
                    #    xywhs_norma = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #print(f"yolo co-ord : {xywhs}")
                    confs = det[:, 4]
                    clss = det[:, 5]
                    print("xywhs:", f"{xywhs.cpu()}")
                    print("confs:", f"{confs.cpu()}")
                    print("clss:", f"{clss.cpu()}")
                    # pass detections to strongsort
                    t4 = time_sync()
                    outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4
                    
                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                            print(f"detection box after tracking: {output}")
                            bboxes = output[0:4]
                            #print(f"yolo deepsort co-ord : {bboxes}")
                            id = output[4]
                            cls = output[5]
                            if isEyeDetection and isGenderClassification:
                                face = im0[int(max(0,bboxes[1])):int(min(bboxes[3],im0.shape[0]-1)),int(max(0,bboxes[0])):int(min(bboxes[2], im0.shape[1]-1))]
                                #TODO 
                                '''
                                Applying gamma correction on face.
                                '''
                                #count += 1
                                face_temp_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                mean = np.mean(face_temp_gray)
                                mid = automate_histogram_mid(mean)
                                gamma = math.log(mid)/math.log(mean/255)
                                # do gamma correction
                                face_gamma = adjust_gamma(face, gamma)
                                #print(face.shape)
                                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                                genderNet.setInput(blob)
                                genderPreds = genderNet.forward()
                                gender = genderList[genderPreds[0].argmax()]
                                genderConf = genderPreds[0].max()
                                #Adding grayscaled face patch to original image
                                if eye_model_type == "cascade":
                                    img0_gamma = im0.copy()
                                    img0_gamma[int(max(0,bboxes[1])):int(min(bboxes[3],im0.shape[0]-1)),int(max(0,bboxes[0])):int(min(bboxes[2], im0.shape[1]-1))] = face_gamma
                                    frame_gray = cv2.cvtColor(img0_gamma, cv2.COLOR_BGR2GRAY)
                                    frame_gray = cv2.equalizeHist(frame_gray)
                                    faceROI = frame_gray[int(max(0,bboxes[1])):int(min(bboxes[3],im0.shape[0]-1)),int(max(0,bboxes[0])):int(min(bboxes[2], im0.shape[1]-1))]
                                    #if True:#dump_gamma_corrected:
                                    #    cv2.imwrite(f"/WS/gamma_corrected/{count}.jpg", face)
                                    #    cv2.imwrite(f"/WS/gamma_corrected/{count}_greyscale.jpg", face_temp_gray)
                                    #    cv2.imwrite(f"/WS/gamma_corrected/{count}_gamma.jpg", face_gamma)
                                    #    cv2.imwrite(f"/WS/gamma_corrected/{count}_org_full.jpg", im0)
                                    #    cv2.imwrite(f"/WS/gamma_corrected/{count}_gamma_full.jpg", img0_gamma)
                                    eyes = eyes_cascade.detectMultiScale(faceROI,1.3, 5)
                                    #print(f"eyes: {eyes}")
                                    if len(eyes):
                                        is_eye_visible = True
                                    else:
                                        is_eye_visible = False
                                    for (x2,y2,w2,h2) in eyes:
                                        eye_center = (int(bboxes[0] + x2 + w2//2), int(bboxes[1] + y2 + h2//2))
                                        radius = int(round((w2 + h2)*0.25))
                                        #print(f"eye_center: {eye_center}")
                                        #print(f"eyes radius: {radius}")
                                        im0 = cv2.circle(im0, eye_center, radius, (255, 255, 0 ), 4)
                                if eye_model_type == "openvino":
                                    im0, pred, is_eye_visible = get_eye_box(bboxes,im0)

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                roi = [polygon_roi[0][0], polygon_roi[0][1],\
                                                     polygon_roi[2][0], polygon_roi[2][1]]
                                #iou_conf = iou_check(bboxes, roi)
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                iou_conf = iou_check(bboxes, roi)
                                #print(f"iou_conf: {iou_conf}")
                                if iou_conf > 0.5:
                                    if USECASE == "Billboard":
                                        annotator.box_label_billboard(bboxes, gender, genderConf, is_eye_visible, label, color=colors(c, True))
                                        if is_eye_visible:
                                            Id = label.strip().split(" ")[0]
                                            if Id not in unique_person_seen_bilboard:
                                                unique_person_seen_bilboard.append(Id)
                                #        billboard_label = f"No. of People who had seen the banner: {len(unique_person_seen_bilboard)}"
                                #        tf = max(annotator.lw - 1, 1)
                                #        w, h = cv2.getTextSize(billboard_label, 0, fontScale=\
                                #                            annotator.lw, thickness=tf*4)[0]
                                #        p1, p2 = (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]))
                                #        p1_ = (0, annotator.im.shape[0]-h)
                                #        outside = p1_[1] - h >= 0
                                #        p2_ = (p1_[0]+w, p1_[1] - h  if outside else p1_[1] + h )
                                #        #p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                                #        cv2.rectangle(annotator.im, p1_, p2_, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                                #        cv2.putText(annotator.im, billboard_label, (p1_[0], p1_[1] - 2 \
                                #                if outside else p1_[1] + h + 2), 0, annotator.lw, txt_color,\
                                #                thickness=tf*4, lineType=cv2.LINE_AA)

                                    if USECASE == "TyreCounting":
                                        tyre_id = f'{id:0.0f}'
                                        if tyre_id not in unique_tyre_centroid:
                                            unique_tyre_centroid = {}
                                            unique_tyre_centroid[tyre_id] = {"trail-path": [], "isLoaded": False}
                                        centroid = int(bboxes[0]) , int((bboxes[1] + bboxes[3])// 2)
                                        unique_tyre_centroid[tyre_id]["trail-path"].append(centroid)
                                        for point in unique_tyre_centroid[tyre_id]["trail-path"]:
                                            print(f"point: {point}")
                                            annotator.im = cv2.circle(annotator.im, point, 5, (255,0,255), thickness=3)
                                        #TODO Line Crossing Logic
                                        '''
                                        If mid poit of tyre's TOP-LEFT and BOTTOM-LEFT cross the drawn line then We can say tyre is 
                                        loaded.
                                        '''
                                        #TODO 
                                        '''
                                        For Futur, We will implement
                                        ENTRY and EXIT Logic 
                                        '''
                                        '''
                                        Equation of line
                                        A(X1, Y1), B(X2, Y2)
                                        '''
                                        if not unique_tyre_centroid[tyre_id]["isLoaded"]:
                                            (X1,Y1), (X2,Y2) = line[0], line[1]
                                            PX,PY = centroid
                                            Z = PX*(Y2-Y1)-PY*(X2-X1)-X1*(Y2-Y1)+Y1*(X2-X1)
                                            if Z <= 0:
                                                #Centroid is on or left side of line, means Tyre is loaded
                                                unique_tyre_centroid[tyre_id]["isLoaded"] = True
                                                annotator.box_label_tyrecounting(bboxes, 1.0,\
                                                                    unique_tyre_centroid[tyre_id]\
                                                                    ["isLoaded"], label, color=colors(c, True))
                                                Id = label.strip().split(" ")[0]
                                                if Id not in unique_tyre:
                                                    unique_tyre.append(Id)
                                        #loading_label = f"No. of Tyre Loaded: {len(unique_tyre)}"
                                        #tf = max(annotator.lw - 1, 1)
                                        #w, h = cv2.getTextSize(loading_label, 0, fontScale=\
                                        #                    annotator.lw, thickness=tf*4)[0]
                                        #p1, p2 = (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]))
                                        #p1_ = (0, annotator.im.shape[0]-h)
                                        #outside = p1_[1] - h >= 0
                                        #p2_ = (p1_[0]+w, p1_[1] - h  if outside else p1_[1] + h )
                                        ##p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                                        #cv2.rectangle(annotator.im, p1_, p2_, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                                        #cv2.putText(annotator.im, loading_label, (p1_[0], p1_[1] - 2 \
                                        #        if outside else p1_[1] + h + 2), 0, annotator.lw, txt_color,\
                                        #        thickness=tf*4, lineType=cv2.LINE_AA)
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                    LOGGER.info(f'Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

                else:
                    strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')
                txt_color=(255, 255, 255)
                if USECASE == "Billboard":
                    #annotator.box_label_billboard(bboxes, gender, genderConf, is_eye_visible, label, color=colors(c, True))
                    #if is_eye_visible:
                    #    Id = label.strip().split(" ")[0]
                    #    if Id not in unique_person_seen_bilboard:
                    #        unique_person_seen_bilboard.append(Id)
                    billboard_label = f"No. of People who had seen the banner: {len(unique_person_seen_bilboard)}"
                    tf = max(annotator.lw - 1, 1)
                    w, h = cv2.getTextSize(billboard_label, 0, fontScale=\
                                        annotator.lw/2, thickness=tf*2)[0]
                    p1_ = (0, annotator.im.shape[0]-h)
                    outside = p1_[1] - h >= 0
                    p2_ = (p1_[0]+w, p1_[1] - h  if outside else p1_[1] + h )
                    #p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(annotator.im, p1_, p2_, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.putText(annotator.im, billboard_label, (p1_[0], p1_[1] - 2 \
                            if outside else p1_[1] + h + 2), 0, annotator.lw/2, txt_color,\
                            thickness=tf*2, lineType=cv2.LINE_AA)
                else:
                    #Id = label.strip().split(" ")[0]
                    #if Id not in unique_tyre:
                    #    if isLoaded:
                    #        unique_tyre.append(Id)

                    loading_label = f"No. of Tyre Loaded: {len(unique_tyre)}"
                    tf = max(annotator.lw - 1, 1)
                    w, h = cv2.getTextSize(loading_label, 0, fontScale=\
                                        annotator.lw, thickness=tf)[0]
                    #p1, p2 = (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3]))
                    p1_ = (0, annotator.im.shape[0]-h)
                    outside = p1_[1] - h >= 0
                    p2_ = (p1_[0]+w, p1_[1] - h  if outside else p1_[1] + h )
                    #p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(annotator.im, p1_, p2_, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.putText(annotator.im, loading_label, (p1_[0], p1_[1] - 2 \
                            if outside else p1_[1] + h + 2), 0, annotator.lw/2, txt_color,\
                            thickness=tf*2, lineType=cv2.LINE_AA)


                                    

                # Stream results
                im0 = annotator.result()
                #if show_vid:
                #    cv2.imshow(str(p), im0)
                #    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_vid:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = int(fps) #vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]
            #Send image to server to imshow
            im0 = cv2.resize(im0, (1920, 1080))
            #im0 = cv2.resize(im0, (1920, 1080),interpolation = cv2.INTER_NEAREST)
            sender.send_image(host_name, im0)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_vid:
            s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--model-version', help='yolov4 or yolov5')
    parser.add_argument('--cfg', default="yolov4-csp-custom.cfg", help='yolov4 cfg file')
    parser.add_argument('--fps', default="5", help='video fps')
    parser.add_argument('--eye-model-type', default="cascade", help='Eye detection model type [cascade, openvino]')
    parser.add_argument('--usecase', help='[Billboard, TyreCounting]')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    #run(**vars(opt))
    run(opt)


if __name__ == "__main__":
    MD = None
    IMG_STRING_TUPLES = None
    RESPONSE = None
    RESPONSE_SENT = False
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-externalconfig', help='External config file path')
    args = parser.parse_args()
    external_config_file = args.externalconfig.strip()
    objYaml = YamlReader(external_config_file)
    objYaml.fetch_config()
    #opt = parse_opt()
    #opt = YamlReader(opt)
    main(objYaml)
