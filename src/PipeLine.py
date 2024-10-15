from typing import Tuple, List, Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results
from src.utils import find_name
# RESIZE

class PipeLine:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.location_model = YOLO(str(os.path.join(self.config["model_dir"], self.config["location_model_name"])))
        self.number_model = YOLO(str(os.path.join(self.config["model_dir"], self.config["digit_model_name"])))
        self.location_size = (640,640)
        self.digit_size = (200,70)

    def predict(self, x: np.array, save=False) -> tuple[List[Tuple[int, str]],np.array]:
        x = cv2.resize(x, self.location_size)
        pred_loc, box = self.predict_location(x)
        cropped_image = self.crop_image(x, box)

        cropped_image = cv2.resize(cropped_image, self.digit_size)
        pred_digits = self.predict_numbers(cropped_image)

        result = self.get_result(pred_digits)
        print(result)

        if save:
            path = os.path.join(self.config["results_dir"], find_name(self.config["results_dir"], "predictions"))
            os.makedirs(path, exist_ok=True)
            plt.imsave(os.path.join(path, "original.jpg"), x)
            pred_digits[0].save(os.path.join(path, "predict_digits.jpg"))
            pred_loc[0].save(os.path.join(path, "predict.jpg"))
            plt.imsave(os.path.join(path, "cropped_image.jpg"), cropped_image)

            self.save_result(os.path.join(path, "result_points.txt"), result)
            self.save_result(os.path.join(path, "result.txt"), [e[1] for e in result])
            print(f"Results saved in {path}")

        return result,pred_digits

    def predict_location(self, x: np.array) -> Tuple[List[Results], Any]:
        pred = self.location_model.predict([x], save=False, conf=self.config["conf"], iou=self.config["iou"], verbose=False)
        return pred, pred[0].boxes.xyxy.tolist()[0]

    def predict_numbers(self, x: np.array):
        pred = self.number_model.predict(x, save=False, conf=self.config["conf"], iou=self.config["iou"], verbose=False)
        return pred

    def draw_box(self, img : np.array, bboxes) :
        for b1 in bboxes :
            label = classes_dict[b1[0]]

            x_center = float(b1[1]) * self.digit_size[0]
            y_center = float(b1[2]) * self.digit_size[1]
            w = float(b1[3]) * self.digit_size[0]
            h = float(b1[4]) * self.digit_size[1]

            x_min = round(x_center - (w / 2))
            x_max = round(x_center + (w / 2))
            y_min = round(y_center - (h / 2))
            y_max = round(y_center + (h / 2))

            x1 = round(x_center - w/4)
            x2 = round(x_center + w/4)
            y1 = round(3)
            y2 = round(10)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1+9), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def get_result(pred: list) -> List[Tuple[int, str]]:
        return [
                (int(item.boxes.xyxy[0][0]), classes_dict[str(int(item.boxes.cls))])
                for item in pred[0]
            ]


    @staticmethod
    def crop_image(x: np.array, box: List[float]) -> np.array:
        return x[round(box[1]):round(box[3]), round(box[0]):round(box[2])]

    @staticmethod
    def load_config(path: str):
        with open(path, "r") as y:
            data = yaml.load(y, Loader=yaml.FullLoader)
        return data

    @staticmethod
    def save_result(path: str, result):
        with open(path, 'w') as f:
            for element in result:
                f.write(f"{element} ")



classes_dict =  {
    '0':'0',
    '1':'1',
    '2':'2',
    '3':'3',
    '4':'4',
    '5':'5',
    '6':'6',
    '7':'7',
    '8':'8',
    '9':'9',
    '10':'B',
    '11':'C',
    '12':'D',
    '13':'G',
    '14':'H',
    '15':'J',
    '16':'L',
    '17':'M',
    '18':'N',
    '19':'S',
    '20':'T',
    '21':'V',
    '22':'Y'
}

