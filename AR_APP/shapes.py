import numpy as np
import cv2

HANDLE_SIZE = 10

class ShapeManager:
    def __init__(self):
        self.shapes = []
        self.selected_idx = -1
        self.mode = 'idle'
        self.drag_start = None
        self.resize_corner = None
        self.current_shape = 'rectangle'

    @staticmethod
    def point_near(p1, p2, radius=HANDLE_SIZE):
        return np.linalg.norm(np.array(p1) - np.array(p2)) <= radius

    @staticmethod
    def get_shape_corners(shape):
        x1, y1, x2, y2 = shape['data']
        return [(x1, y1), (x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2)]

    def draw_shape(self, img, shape, selected=False):
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
            for corner in self.get_shape_corners(shape):
                cv2.circle(img, corner, HANDLE_SIZE, (255, 255, 255), -1)

    def point_in_shape(self, point, shape):
        x, y = point
        x1, y1, x2, y2 = shape['data']
        return x1 <= x <= x2 and y1 <= y <= y2

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, shape in enumerate(self.shapes):
                for corner in self.get_shape_corners(shape):
                    if self.point_near((x, y), corner):
                        self.selected_idx = i
                        self.resize_corner = corner
                        self.mode = 'resizing'
                        return
            for i, shape in enumerate(self.shapes):
                x1, y1, x2, y2 = shape['data']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_idx = i
                    self.drag_start = (x, y)
                    self.mode = 'moving'
                    return
            self.drag_start = (x, y)
            self.selected_idx = -1
            self.mode = 'drawing'

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == 'moving' and self.selected_idx != -1:
                dx, dy = x - self.drag_start[0], y - self.drag_start[1]
                x1, y1, x2, y2 = self.shapes[self.selected_idx]['data']
                self.shapes[self.selected_idx]['data'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                self.drag_start = (x, y)
            elif self.mode == 'resizing' and self.selected_idx != -1:
                x1, y1, x2, y2 = self.shapes[self.selected_idx]['data']
                # Redimensionnement simple (à améliorer)
                self.shapes[self.selected_idx]['data'] = (x, y, x2, y2)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == 'drawing' and self.drag_start:
                self.shapes.append({'type': self.current_shape, 'data': (*self.drag_start, x, y)})
            self.mode = 'idle'

        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, shape in enumerate(self.shapes):
                x1, y1, x2, y2 = shape['data']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    del self.shapes[i]
                    self.selected_idx = -1
                    break
