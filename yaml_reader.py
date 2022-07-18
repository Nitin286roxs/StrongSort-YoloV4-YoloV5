import sys
import yaml

class YamlReader:
    def __init__(self, file_path):
        self.objyaml = None
        with open(file_path, "r") as args:
            try:
                self.objyaml = yaml.safe_load(args)
            except yaml.YAMLError as exc:
                print(exc)
 
        self.yolo_weights = None
        self.strong_sort = "osnet_x0_25_market1501.pt"
        self.source = None
        self.conf_thres = 0.1
        self.iou_thres = 0.1
        self.save_vid = True 
        self.model_version = "yolov4"
        self.cfg = "yolov4/models/yolov4-csp-custom.cfg"
        self.device = 0
        self.fps = 25
        self.eye_model_type = None
        self.usecase = None
        self.imgsz=(640, 640)  # inference size (height, width)
        self.max_det=1000  # maximum detections per image
        self.show_vid=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project='runs/track'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.hide_class=False  # hide IDs
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference

    def fetch_config(self):
        if 'yolo-weights' in self.objyaml:
            self.yolo_weights = self.objyaml['yolo-weights']
        if 'strong-sort' in self.objyaml:
            self.strong_sort = self.objyaml['strong-sort']
        if 'source' in self.objyaml:
            self.source = self.objyaml['source']
        if 'conf-thres' in self.objyaml:
            self.conf_thres = self.objyaml['conf-thres']
        if 'iou-thres' in self.objyaml:
            self.iou_thres = self.objyaml['iou-thres']
        if 'conf-thres' in self.objyaml:
            self.conf_thres = self.objyaml['conf-thres']
        if 'save-vid' in self.objyaml:
            self.save_vid = self.objyaml['save-vid']
        if 'model-version' in self.objyaml:
            self.model_version = self.objyaml['model-version']
        if 'cfg' in self.objyaml:
            self.cfg = self.objyaml['cfg']
        if 'device' in self.objyaml:
            self.device = self.objyaml['device']
        if 'fps' in self.objyaml:
            self.fps = self.objyaml['fps']
        if 'usecase' in self.objyaml:
            self.usecase = self.objyaml['usecase']
        if 'eye-model-type' in self.objyaml:
            self.eye_model_type = self.objyaml['eye-model-type']
        if 'classes' in self.objyaml:
            self.classes = self.objyaml['classes']

    def print_config(self):
        print(f"self.yolo_weights: {self.yolo_weights}")
        print(f"self.strong_sort: {self.strong_sort}")
        print(f"self.source: {self.source}")
        print(f"self.conf_thres: {self.conf_thres}")
        print(f"self.iou_thres: {self.iou_thres}")
        print(f"self.save_vid: {self.save_vid}")
        print(f"self.model_version: {self.model_version}")
        print(f"self.cfg: {self.cfg}")
        print(f"self.device: {self.device}")
        print(f"self.fps: {self.fps}")

#file_path = sys.argv[1]
#objYaml = YamlReader(file_path)
#objYaml.fetch_config()
#objYaml.print_config()

