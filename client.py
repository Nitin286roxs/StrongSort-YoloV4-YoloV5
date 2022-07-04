import os
import sys
import json
import time
import cv2
import numpy as np
import requests
import base64

DATA = {}#json.load(open(sys.argv[1], "r"))
print("Connecting to Server ...")
stream = sys.argv[1]

vcap = cv2.VideoCapture(stream)
cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
request = 0
while(1):
    ret, IMG = vcap.read()
    if ret == False:
        print("Frame is empty")
        break;
    else:
        #cv2.imshow('VIDEO', IMG)
        #cv2.waitKey(1)
        IMAGE_STRING = base64.b64encode(cv2.imencode('.bmp', IMG)[1]).decode("utf-8")
        print("Sending request {}".format(request))
        MD = {"metadata": {"sender": "Hello "}, "images": [{"image_string": IMAGE_STRING, "len": len(IMAGE_STRING)}]}
        #response = requests.post('https://10.16.239.1:5011/ovc_input', json=MD)
        START = time.time()*1000
        #response =  requests.post('http://172.17.0.3:5011/process', json=MD)
        response = requests.post('http://127.0.0.1:5011/process', json=MD)
        #response = requests.post('http://0.0.0.0:5011/process', json=MD)
        END = time.time()*1000
        print("time taken for request ", request, ": ", END-START, " msec.")
        print(response.status_code)
        result_json = json.loads(response.content.decode('utf-8'))
        print(result_json)
        #file_.write(IMGPATH + str(result_json["moduleData"]["imageList"][0]["inference"])+"\n")
        print("Send completed")
        request += 1
