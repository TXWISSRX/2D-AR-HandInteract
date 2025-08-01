import cv2
from ultralytics import YOLO
import mediapipe as mp

from AR_APP.shapes import ShapeManager
from AR_APP.detection import detect_hands, mp_hands
from AR_APP.overlay import overlay_mask, extract_object, overlay_transparent
from AR_APP.utils import compute_iou_area
import numpy as np

def main():
    shape_manager = ShapeManager()
    model = YOLO("yolo11m-seg.pt")
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Camera error")
        return

    cv2.namedWindow("YOLO Editor")
    cv2.setMouseCallback("YOLO Editor", shape_manager.mouse_event)

    captured_object = None
    is_holding = False
    was_holding = False
    released_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        results = model.predict(source=frame, task='segment', verbose=False)
        result = results[0]

        pt_thumb, pt_index = detect_hands(hands, frame)
        if pt_thumb:
            cv2.circle(display, pt_thumb, 10, (0, 255, 255), -1)
        if pt_index:
            cv2.circle(display, pt_index, 10, (255, 0, 255), -1)

        if result.boxes is not None and result.masks is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names
            masks = result.masks.data.cpu().numpy()

            for i, shape in enumerate(shape_manager.shapes):
                shape_manager.draw_shape(display, shape, selected=(i == shape_manager.selected_idx))

                x1, y1, x2, y2 = shape['data']
                user_area = abs((x2 - x1) * (y2 - y1))
                best_match = None
                best_score = float('inf')

                for j, box in enumerate(boxes):
                    x, y, w_, h_ = box
                    bx1, by1, bw, bh = int(x - w_ / 2), int(y - h_ / 2), int(w_), int(h_)
                    box_area = bw * bh
                    iou, inter_area = compute_iou_area((x1, y1, x2, y2), [bx1, by1, bw, bh])
                    ioua = inter_area / user_area if user_area > 0 else 0

                    if ioua > 0.5 and abs(box_area - user_area) < best_score:
                        best_score = abs(box_area - user_area)
                        best_match = (names[int(classes[j])], j)

                if best_match:
                    label, idx = best_match
                    if label.lower() != "person":
                        is_thumb_inside = pt_thumb and shape_manager.point_in_shape(pt_thumb, shape)
                        is_index_inside = pt_index and shape_manager.point_in_shape(pt_index, shape)
                        mask_color = (255, 0, 0) if is_index_inside and is_thumb_inside else (0, 255, 0)
                        display = overlay_mask(display, masks[idx], color=mask_color)

                        if is_thumb_inside and is_index_inside and pt_index is not None and pt_thumb is not None:
                            dist = np.linalg.norm(np.array(pt_index) - np.array(pt_thumb))
                            if dist < 50:
                                if not is_holding:
                                    captured_object = extract_object(masks[idx], frame)
                                is_holding = True
                        elif pt_index is not None and pt_thumb is not None:
                            dist = np.linalg.norm(np.array(pt_index) - np.array(pt_thumb))
                            if dist >= 50:
                                is_holding = False

        if was_holding and not is_holding and captured_object is not None and pt_index is not None:
            released_objects.append((captured_object.copy(), pt_index))
            captured_object = None

        was_holding = is_holding

        if is_holding and captured_object is not None and pt_index is not None:
            display = overlay_transparent(display, captured_object, *pt_index)

        for obj_img, center in released_objects:
            display = overlay_transparent(display, obj_img, *center)

        cv2.putText(display, f"Mode: {shape_manager.current_shape}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("YOLO Editor", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            shape_manager.current_shape = 'rectangle'
        elif key == ord('c'):
            shape_manager.current_shape = 'circle'
        elif key == ord('e'):
            shape_manager.current_shape = 'ellipse'
        elif key == 27:  # ESC
            shape_manager.shapes.clear()
            shape_manager.selected_idx = -1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
