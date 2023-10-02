import cv2
from ultralytics import YOLO
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = anns

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


# model = YOLO("yolov8m.pt").to(torch.device("mps"))
cap = cv2.VideoCapture(0)

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "mps"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device=device).to(torch.float)
sam = torch.compile(sam)

# mask_generator = SamAutomaticMaskGenerator(sam)

# masks = mask_generator.generate(sample_image)
predictor = SamPredictor(sam)

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        shrink_factor = 0.1
        frame = cv2.resize(frame, None, fx=shrink_factor, fy=shrink_factor)
        predictor.set_image(frame)
        point_y, point_x = frame.shape[0] // 2, frame.shape[1] // 2
        input_point = np.array([[point_x, point_y]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        # masks = masks.transpose(1, 2, 0)
        masks = masks.squeeze(0)
        # masks = np.broadcast_to(masks, (masks.shape[0], masks.shape[1], 3))
        mask_color = np.array([0, 200, 0])
        frame[masks] = 0.5 * frame[masks] + 0.5 * mask_color
        frame = cv2.circle(frame, (point_x, point_y), color=(255, 0, 0), radius=5)

        if not ret:
            break
        # frame = cv2.resize(frame, None, fx=1/shrink_factor, fy=1/shrink_factor)
        cv2.imshow("Img", frame)
        print("showing image")
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()

