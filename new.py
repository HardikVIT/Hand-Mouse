# geometric_gestures.py

import cv2
import time
import mediapipe as mp
import math
import pyautogui

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

prev_distance = None
hold_start = 0

def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def get_landmarks(results, img_shape):
    lmList = []
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        h, w, _ = img_shape
        for id, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((cx, cy))
    return lmList

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lmList = get_landmarks(results, img.shape)

    if lmList:
        x1, y1 = lmList[4]   # Thumb tip
        x2, y2 = lmList[8]   # Index tip
        x3, y3 = lmList[12]  # Middle tip

        draw.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # ----- Click Gesture -----
        if distance((x1, y1), (x2, y2)) < 30:
            cv2.putText(img, "Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            pyautogui.click()

        # ----- Scroll Gesture -----
        if abs(y2 - y3) < 30:  # both fingers aligned vertically
            scroll_amount = y2 - y3
            pyautogui.scroll(-scroll_amount)
            cv2.putText(img, "Scroll", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2)

        # ----- Zoom Gesture -----
        curr_distance = distance((x1, y1), (x2, y2))
        if prev_distance:
            if curr_distance > prev_distance + 15:
                cv2.putText(img, "Zoom In", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            elif curr_distance < prev_distance - 15:
                cv2.putText(img, "Zoom Out", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        prev_distance = curr_distance
        # ----- Hold Gesture -----
        if distance((x1, y1), (x2, y2)) < 30:
            if hold_start == 0:
                hold_start = time.time()
            elif time.time() - hold_start > 1:
                cv2.putText(img, "Hold", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 2)
        else:
            hold_start = 0

    cv2.imshow("Hand Gesture Controller", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
