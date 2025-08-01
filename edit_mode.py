import cv2
import numpy as np
from ultralytics import YOLO

# Global state
drawing = False
moving = False
resizing = False
selected_idx = -1
start_point = None
shapes = []  # list of dicts: {type: 'rectangle'/'ellipse'/'circle', data: (x1, y1, x2, y2)}
mode = 'idle'
drag_start = None
resize_corner = None
HANDLE_SIZE = 10
current_shape = 'rectangle'  # default shape type



def point_near(p1, p2, radius=HANDLE_SIZE):
    return np.linalg.norm(np.array(p1) - np.array(p2)) <= radius


def get_shape_corners(shape):
    x1, y1, x2, y2 = shape['data']
    return [(x1, y1), (x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2)]


def draw_shape(img, shape, selected=False):
    x1, y1, x2, y2 = shape['data']
    if shape['type'] == 'rectangle':
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    elif shape['type'] == 'circle':
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = int(np.hypot(x2 - x1, y2 - y1) / 2)
        cv2.circle(img, center, radius, (255, 255, 0), 2)
    elif shape['type'] == 'ellipse':
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (abs(x2 - x1) // 2, abs(y2 - y1) // 2)
        cv2.ellipse(img, center, axes, 0, 0, 360, (0, 255, 255), 2)

    if selected:
        for corner in get_shape_corners(shape):
            cv2.circle(img, corner, HANDLE_SIZE, (255, 255, 255), -1)


def compute_iou(rect, bbox):
    rx1, ry1, rx2, ry2 = rect
    bx1, by1, bw, bh = bbox
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(rx1, bx1)
    inter_y1 = max(ry1, by1)
    inter_x2 = min(rx2, bx2)
    inter_y2 = min(ry2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    rect_area = (rx2 - rx1) * (ry2 - ry1)
    return inter_area / rect_area


def overlay_mask(frame, mask, alpha=0.4, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    bin_mask = (resized > 0.5).astype(np.uint8)
    overlay = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[:, :, :] = color
    return np.where(bin_mask[:, :, None], cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0), frame)


def mouse_callback(event, x, y, flags, param):
    global drawing, mode, start_point, selected_idx, drag_start, resize_corner, shapes

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, shape in enumerate(shapes):
            for corner in get_shape_corners(shape):
                if point_near((x, y), corner):
                    selected_idx = i
                    resize_corner = corner
                    mode = 'resizing'
                    return
        for i, shape in enumerate(shapes):
            x1, y1, x2, y2 = shape['data']
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_idx = i
                drag_start = (x, y)
                mode = 'moving'
                return
        start_point = (x, y)
        selected_idx = -1
        mode = 'drawing'

    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == 'moving' and selected_idx != -1:
            dx, dy = x - drag_start[0], y - drag_start[1]
            x1, y1, x2, y2 = shapes[selected_idx]['data']
            shapes[selected_idx]['data'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            drag_start = (x, y)
        elif mode == 'resizing' and selected_idx != -1:
            x1, y1, x2, y2 = shapes[selected_idx]['data']
            shapes[selected_idx]['data'] = (x, y, x2, y2)

    elif event == cv2.EVENT_LBUTTONUP:
        if mode == 'drawing' and start_point:
            shapes.append({'type': current_shape, 'data': (*start_point, x, y)})
        mode = 'idle'

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, shape in enumerate(shapes):
            x1, y1, x2, y2 = shape['data']
            if x1 <= x <= x2 and y1 <= y <= y2:
                del shapes[i]
                selected_idx = -1
                break


def main():
    global current_shape, selected_idx

    model = YOLO("yolo11m-seg.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    cv2.namedWindow("YOLO Editor")
    cv2.setMouseCallback("YOLO Editor", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        results = model.predict(source=frame, task='segment', verbose=False)
        result = results[0]

        if result.boxes is not None and result.masks is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names
            masks = result.masks.data.cpu().numpy()

            for i, shape in enumerate(shapes):
                draw_shape(display, shape, selected=(i == selected_idx))
                x1, y1, x2, y2 = shape['data']
                user_area = abs((x2 - x1) * (y2 - y1))
                best_match = None
                best_score = float('inf')

                for j, box in enumerate(boxes):
                    x, y, w, h = box
                    bx1, by1, bw, bh = int(x - w / 2), int(y - h / 2), int(w), int(h)
                    box_area = bw * bh
                    iou = compute_iou((x1, y1, x2, y2), [bx1, by1, bw, bh])
                    if iou > 0.1 and abs(box_area - user_area) < best_score:
                        best_score = abs(box_area - user_area)
                        best_match = (names[int(classes[j])], j)

                if best_match:
                    label, idx = best_match
                    display = overlay_mask(display, masks[idx])
                    

        cv2.putText(display, f"Mode: {current_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("YOLO Editor", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_shape = 'rectangle'
        elif key == ord('c'):
            current_shape = 'circle'
        elif key == ord('e'):
            current_shape = 'ellipse'
        elif key == 27:
            shapes.clear()
            selected_idx = -1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()