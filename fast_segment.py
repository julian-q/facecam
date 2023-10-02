import cv2
from ultralytics import YOLO, SAM
import numpy as np
import torch

# model = YOLO("yolov8m.pt").to(torch.device("mps"))
model = SAM("sam_b.pt").to(torch.device("mps"))
model = torch.compile(model)
cap = cv2.VideoCapture(0)

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        # results = model(frame)
        # result = results[0]
        # bboxes = result.boxes.xyxy
        # bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # classes = np.array(result.boxes.cls.cpu(), dtype="int")
        # for cls, bbox in zip(classes, bboxes):
        #     (x, y, x2, y2) = bbox
        #     cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        #     cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        results = model(frame, points=[500, 500], labels=[1])
        breakpoint()


        if not ret:
            break
        cv2.imshow("Img", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()

