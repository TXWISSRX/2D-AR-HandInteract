import cv2
import numpy as np
import mediapipe as mp

THUMB_TIP = 4
INDEX_TIP = 8

mp_hands = mp.solutions.hands

def detect_hands(hands_model, frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb_frame)
    pt_thumb = pt_index = None
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            thumb = hand.landmark[THUMB_TIP]
            index = hand.landmark[INDEX_TIP]
            pt_thumb = (int(thumb.x * w), int(thumb.y * h))
            pt_index = (int(index.x * w), int(index.y * h))
    return pt_thumb, pt_index
