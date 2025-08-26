"""
utils.py
Utilities for:
- loading Ultralytics YOLO model
- running detection
- computing severity from bounding boxes (bbox union) or refined GrabCut masks
- drawing overlay visualization

Requires:
    pip install ultralytics opencv-python pillow numpy
"""

from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple

# -------------------------
# Helpers / Configuration
# -------------------------
# Map class IDs to readable names. Replace with your model's names if different.
# If ultralytics model.names is accessible, we'll use it; this is fallback.
DEFAULT_NAMES = {
    0: "healthy",
    1: "curl_stage1",
    2: "curl_stage2",
    3: "sooty_mold",
    4: "leaf_enation",
}

def load_yolo_model(model_path: str):
    """
    Load ultralytics YOLO model. Returns model object.
    """
    model = YOLO(model_path)
    # model.names available
    return model

# -------------------------
# Detection wrapper
# -------------------------
def detect_image(model, image: np.ndarray, conf: float = 0.25, iou: float = 0.45):
    """
    Run YOLO detection on a numpy image (H,W,3).
    Returns:
        detections: list of dict { 'xyxy': (xmin,ymin,xmax,ymax), 'conf': float, 'cls': int, 'name': str }
    """
    # Ultralytics new API: results = model.predict(source=image, conf=conf, iou=iou)
    # but calling model(image, conf=...) also works and returns Results
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    r = results[0]
    detections = []
    # Boxes accessible as r.boxes: has xyxy, conf, cls
    try:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
        confs = boxes.conf.cpu().numpy()  # (N,)
        clss = boxes.cls.cpu().numpy().astype(int)  # (N,)
    except Exception:
        # If no detections, return empty
        return []
    # try to obtain names from model if possible
    model_names = getattr(model, "model", None)
    # fallback mapping
    names_map = getattr(model, "names", DEFAULT_NAMES)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        conf = float(confs[i])
        cls = int(clss[i])
        name = names_map.get(cls, str(cls))
        detections.append({"xyxy": (int(x1), int(y1), int(x2), int(y2)), "conf": conf, "cls": cls, "name": name})
    return detections

# -------------------------
# Severity computation
# -------------------------
def union_mask_from_boxes(image_shape: Tuple[int,int], boxes: List[Tuple[int,int,int,int]]):
    """
    Create a binary mask where union of all boxes is 1. image_shape = (H,W)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w-1, int(x2)); y2 = min(h-1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2+1, x1:x2+1] = 1
    return mask

def compute_severity_from_boxes(image: np.ndarray, detections: List[Dict], method: str = "bbox") -> Dict:
    """
    Compute severity metrics.
    Args:
        image: np.ndarray (H,W,3)
        detections: list of dict from detect_image
        method: "bbox" (fast) or "grabcut" (refined)
    Returns:
        dict {
          "per_box": [ {"name":.., "ratio":.., "box":(...) , "conf":.. }, ... ],
          "overall_ratio": float,
          "overall_label": "mild"/"moderate"/"severe",
        }
    """
    H, W = image.shape[:2]
    img_area = H * W
    boxes = [d["xyxy"] for d in detections]
    # if no detections
    if len(boxes) == 0:
        return {"per_box": [], "overall_ratio": 0.0, "overall_label": "none"}

    if method == "bbox":
        # union area of boxes divided by image area
        union_mask = union_mask_from_boxes((H,W), boxes)
        union_area = int(np.count_nonzero(union_mask))
        overall_ratio = union_area / img_area
        per_box = []
        for d in detections:
            x1,y1,x2,y2 = d["xyxy"]
            box_area = max(0, int((x2 - x1) * (y2 - y1)))
            per_box.append({"name": d["name"], "box_area": box_area, "ratio": box_area / img_area, "box":d["xyxy"], "conf": d["conf"]})
    elif method == "grabcut":
        # refine each box using GrabCut, compute mask union
        full_union_mask = np.zeros((H,W), dtype=np.uint8)
        per_box = []
        for d in detections:
            x1,y1,x2,y2 = d["xyxy"]
            # add margin to rectangle (clamp to image)
            pad = 5
            rx1 = max(0, x1-pad); ry1 = max(0, y1-pad)
            rx2 = min(W-1, x2+pad); ry2 = min(H-1, y2+pad)
            # run grabcut on crop
            crop = image[ry1:ry2+1, rx1:rx2+1].copy()
            if crop.size == 0:
                mask = np.zeros((ry2-ry1+1, rx2-rx1+1), dtype=np.uint8)
            else:
                mask = run_grabcut(crop)
            # paste mask back
            full_union_mask[ry1:ry2+1, rx1:rx2+1] = np.logical_or(full_union_mask[ry1:ry2+1, rx1:rx2+1], mask).astype(np.uint8)
            lesion_area = int(np.count_nonzero(mask))
            per_box.append({"name": d["name"], "lesion_area": lesion_area, "ratio": lesion_area / img_area, "box": d["xyxy"], "conf": d["conf"]})
        overall_ratio = int(np.count_nonzero(full_union_mask)) / img_area
    else:
        raise ValueError("method must be 'bbox' or 'grabcut'")

    # severity bucket mapping (tune as necessary)
    if overall_ratio < 0.05:
        overall_label = "mild"
    elif overall_ratio < 0.20:
        overall_label = "moderate"
    else:
        overall_label = "severe"

    return {"per_box": per_box, "overall_ratio": overall_ratio, "overall_label": overall_label}

# -------------------------
# GrabCut helper
# -------------------------
def run_grabcut(bgr_crop: np.ndarray, iterCount: int = 5) -> np.ndarray:
    """
    Apply GrabCut on a small crop to attempt to isolate foreground lesion.
    Input crop: BGR image.
    Returns a binary mask (H,W) where 1 indicates foreground.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return np.zeros((0,0), dtype=np.uint8)

    h, w = bgr_crop.shape[:2]
    rect = (1,1, max(1,w-2), max(1,h-2))  # inside crop
    mask = np.zeros((h,w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(bgr_crop, mask, rect, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT)
        bin_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    except Exception:
        # fallback: simple threshold on saturation channel
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1]
        _, bin_mask = cv2.threshold(s, 40, 1, cv2.THRESH_BINARY)
    return bin_mask

# -------------------------
# Visualization
# -------------------------
def draw_overlay(image: np.ndarray, detections: List[Dict], severity_result: Dict, show_conf=True):
    """
    Draw boxes, labels, and severity bars on a copy of the image and return it.
    """
    out = image.copy()
    H,W = image.shape[:2]
    # draw boxes
    for item in detections:
        x1,y1,x2,y2 = item["xyxy"]
        color = (0,0,255)  # red
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f"{item['name']}:{item['conf']:.2f}" if show_conf else item['name']
        cv2.putText(out, label, (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # show overall severity text
    text = f"Severity: {severity_result.get('overall_label','unknown')} ({severity_result.get('overall_ratio',0.0):.2%})"
    cv2.putText(out, text, (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,200,10), 2, cv2.LINE_AA)
    return out
