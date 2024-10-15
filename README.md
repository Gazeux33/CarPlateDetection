# Car Plate Detection and Recognition

The aim of this project is to create a **pipeline** with two models to detect and recognize car plates. The first model 
is able to detect the car plate in an image and the second model is able to recognize the characters in the car plate.
Theses models are **finetunned Yolov8**.

## PipeLine

<div>
	<img src="https://github.com/Gazeux33/CarPlateDetection/blob/master/assets/pipeline.png">
</div>


## Data

To train theses two models I used two differents dataset.

Dataset for the Location Model : [Car Plate Detection - YoloV8](https://www.kaggle.com/datasets/nimapourmoradi/car-plate-detection-yolov8)
<br>
Dataset for the Digits Model : [Persian Plates Digits Detection - YoloV8](https://www.kaggle.com/code/nimapourmoradi/persian-plates-digits-detection-yolov8)

## How to use
The configuration file is **config.yaml**

Install all dependensies
```bash
pip install -r requirements.txt

```

To process an image you can use this command 
```bash
python main.py path_to_your_image

```


