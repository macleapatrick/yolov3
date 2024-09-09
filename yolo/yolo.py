import cv2
from model import Yolov3
from loss import YOLOv3Loss
from labels import LABELS as keys
from anchors import ANCHORS
from utils import scale_and_normalize
import numpy
import torch
import time

class Yolo():
    # container class for initalizing and running yolo model
    def __init__(self, weights_path='.\yolov3.weights'):
        self.model = Yolov3(weights_path)
        self.model.load_weights()
        self.model.eval()
        self.keys = keys

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def run(self, image, bb_size=1):
        start = time.time()
        image2process, _ = scale_and_normalize(image)
        image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGBA2RGB)
        height, width, _ = image.shape

        image2process = image2process.to(self.device)
        print(f'Preprocessing time: {time.time()-start:.2f}')
        start = time.time()

        height_sf = height / 416
        width_sf = width / 416

        # run model
        with torch.no_grad():
            y1, y2, y3 = self.model(image2process.unsqueeze(0))
            loss_fn = YOLOv3Loss(ANCHORS)
            print(f'Prediction time: {time.time()-start:.2f}')
            start = time.time()

            # transform predictions
            p1 = loss_fn.transform_predictions(y1)
            p2 = loss_fn.transform_predictions(y2)
            p3 = loss_fn.transform_predictions(y3)

            # NMS and convert to bounding boxes
            p = (p1, p2, p3)
            prediction = self.model.convert2bb(p)
            print(f'NMS time: {time.time()-start:.2f}')
            starts = time.time()

        if prediction is not None:
            for p in prediction:

                start = (int(p[0] * width_sf), int(p[1] * height_sf)) 
                end = (int(p[2] * width_sf), int(p[3] * height_sf))
                obj = int(p[5])
                cc = float(p[6])

                image = cv2.rectangle(image, start, end, (0, 255, 255), 4*bb_size)
                caption = f'{self.keys[obj]} {cc:.2f}'
                image = cv2.putText(image, caption, (start[0]+int(10*width_sf), end[1]-int(10*height_sf)), cv2.FONT_HERSHEY_SIMPLEX, 1*bb_size, (255, 255, 0), 2*bb_size, cv2.LINE_AA)

        print(f'Draw time: {time.time()-starts:.2f}')

        return image