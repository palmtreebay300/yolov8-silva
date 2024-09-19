import random

import cv2
import numpy as np
from ultralytics import YOLO

# opening the file in read mode
my_file = open(r"E:\Documents\Python_scripts\yolov8\yolov8-silva\utils\coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Read the image
image = cv2.imread(r"E:\Documents\Python_scripts\yolov8\yolov8-silva\inference\images\img0.JPG", cv2.IMREAD_COLOR)

# Check if the image opened properly
if(image is None):
    print("Could not find or open the image")

# Predict on image
detect_params = model.predict(source=[image], conf=0.45, save=False)

# Convert tensor array to numpy
DP = detect_params[0].numpy()
print(DP)

if len(DP) != 0:
    for i in range(len(detect_params[0])):
        print(i)

        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        cv2.rectangle(
            image,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            2
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            image,
            class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2
        )

    # Display the resulting image
    cv2.imshow("ObjectDetection", image)
    cv2.waitKey(0)

# When everything done, release the capture
cv2.destroyAllWindows()