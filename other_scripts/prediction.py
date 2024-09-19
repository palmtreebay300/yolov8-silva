# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# # Run batched inference on a list of images
# # results = model(["E:\Downloads\objects.jpg", "E:\Documents\Python_scripts\yolov8\yolov8-silva\inference\images\img0.JPG"])  # return a list of Results objects
# results = model("https://youtu.be/QC2DOGdYcNo?si=veQpa6CBE8yI6ghv", stream=True)

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     # result.save(filename="E:\Pictures\Saved Pictures\result.jpg")  # save to disk

from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Define source as YouTube video URL
source = "https://youtu.be/LNwODJXcvt4"

# Run inference on the source
results = model(source, stream=True, show=True)  # generator of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename="E:\Pictures\Saved Pictures\result.jpg")  # save to disk
