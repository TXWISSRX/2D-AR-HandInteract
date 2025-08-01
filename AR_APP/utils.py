def compute_iou_area(rect, bbox):
    rx1, ry1, rx2, ry2 = rect
    bx1, by1, bw, bh = bbox
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(rx1, bx1)
    inter_y1 = max(ry1, by1)
    inter_x2 = min(rx2, bx2)
    inter_y2 = min(ry2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0, 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (rx2 - rx1) * (ry2 - ry1) + bw * bh - inter_area
    return inter_area / union_area, inter_area
