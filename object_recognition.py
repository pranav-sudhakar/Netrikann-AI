import cv2

from ultralytics import YOLO
model = YOLO('yolov8n.pt')

with open('coco.names', 'r') as file:
    lines = file.readlines()

classes = []

for line in lines:
    class_name = line.strip()
    classes.append(class_name)


def recognize_images(filename):
    detected_objects = []
    results = model(filename,show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            clss = int(box.cls[0])
            detected_objects.append(str(classes[clss]))
    return detected_objects