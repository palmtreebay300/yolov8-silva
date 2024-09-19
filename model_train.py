from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")  # pass any model type
results = model.train(data="coco8.yaml", epochs=5)