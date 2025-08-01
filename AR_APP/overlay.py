import cv2
import numpy as np

def overlay_mask(frame, mask, alpha=0.4, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    bin_mask = (resized > 0.5).astype(np.uint8)
    overlay = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[:, :, :] = color
    return np.where(bin_mask[:, :, None], cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0), frame)

def extract_object(mask, frame):
    h, w = frame.shape[:2]
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    bin_mask = (resized_mask > 0.5).astype(np.uint8)
    x, y, w_box, h_box = cv2.boundingRect(bin_mask)
    extracted_rgb = frame[y:y + h_box, x:x + w_box]
    mask_cropped = bin_mask[y:y + h_box, x:x + w_box]
    alpha = (mask_cropped * 255).astype(np.uint8)
    rgba = cv2.cvtColor(extracted_rgb, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba

def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]
    x1, y1 = max(0, x - ow // 2), max(0, y - oh // 2)
    x2, y2 = min(bw, x1 + ow), min(bh, y1 + oh)
    overlay_cropped = overlay[0:(y2 - y1), 0:(x2 - x1)]
    alpha = overlay_cropped[:, :, 3] / 255.0
    for c in range(3):
        background[y1:y2, x1:x2, c] = (1 - alpha) * background[y1:y2, x1:x2, c] + alpha * overlay_cropped[:, :, c]
    return background
