from ultralytics import YOLO
model = YOLO("best.pt")

print(model)
print("Done")