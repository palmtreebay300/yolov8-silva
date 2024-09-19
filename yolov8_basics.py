from ultralytics import YOLO
import numpy as np
import os
from dataPath import DATA_PATH

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

# predict on an image
# source = os.path.join(DATA_PATH,"bird.jpeg")
source = "E:\Documents\Python_scripts\yolov8\yolov8-silva\inference\images\img0.JPG"
detection_output = model.predict(source, conf=0.25, save=True) 

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())