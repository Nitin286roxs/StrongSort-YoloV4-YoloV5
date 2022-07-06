import cv2
import imagezmq
image_hub = imagezmq.ImageHub(open_port='tcp://127.0.0.1:5555', REQ_REP=False)
cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
while True:  # show streamed images until Ctrl-C
    host_name, image = image_hub.recv_image()
    cv2.imshow('VIDEO', image) # 1 window for each RPi
    cv2.waitKey(1)
