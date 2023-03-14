import argparse
import os
import platform
import sys
from pathlib import Path
import numpy
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox


class YOLOv5Detector():
    def __init__(self, weights, device):
        self.weights = weights  # model.pt path(s)
        self.data = './data/coco128.yaml'  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        # Load data
        self.bs = 1  # batch_size
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup

    def detect(self, image, image_name):
        with torch.no_grad():
            seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
            im0s = image
            im = letterbox(im0s, self.imgsz, self.stride, self.pt)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = numpy.ascontiguousarray(im)
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            pred = self.model(im, augment=self.augment, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3
            # print('Image %s done. Preprocess %.3fs, Inference %.3fs, NMS %.3fs' % (image_name, dt[0], dt[1], dt[2]))
            # Process predictions
            det = pred[0]  # Only 1 image
            if len(det):
                # Rescale boxes from img_size to im0s size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                for idx in reversed(range(len(det))):
                    class_id = int(det[idx][5])
                    if class_id != 0:  # Only humans
                        det = det[torch.arange(det.size(0)) != idx]
            return det


# if __name__ == '__main__':
#     source = './data/8k2'
#     output = './runs/hd'
#     image_ids = sorted(os.listdir(source))
#     yolov5_detector = YOLOv5Detector()
#     t_s = time_sync()
#     for i in range(len(image_ids)):
#         image = cv2.imread(os.path.join(source, image_ids[i]))
#         det = yolov5_detector.detect(image, image_ids[i])
#         # Save image
#         save_path = os.path.join(output, image_ids[i])
#         image_res = image.copy()
#         annotator = Annotator(image_res, line_width=3, example=str(yolov5_detector.model.names))
#         for *xyxy, conf, cls in reversed(det):
#             c = int(cls)  # integer class
#             label = f'{yolov5_detector.model.names[c]} {conf:.2f}'
#             annotator.box_label(xyxy, label, color=colors(c, True))
#         cv2.imwrite(save_path, image_res)
#     t_e = time_sync()
#     print('Total time %.3fs' % (t_e - t_s))


