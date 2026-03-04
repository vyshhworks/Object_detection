from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Run detection on an image
results = model("test.jpg", show=True)

# Save the output image
results[0].save(filename="output.jpg")