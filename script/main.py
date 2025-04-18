import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize mediapipe and setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
draw = mp.solutions.drawing_utils

# Setup camera and screen
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Scroll and annotation memory
prev_x, prev_y = None, None
prev_scroll_y = None  # <-- add this line

click_cooldown = 0
smooth_factor = 0.3  # Higher smooth factor for smoother movement

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def fingers_up(landmarks):
    """
    Determine if fingers are up (1) or down (0).
    Based on y-position comparison with lower joints.
    """
    fingers = []

    # Thumb: check if tip x is to the right of IP joint (for right hand)
    fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)

    # Index
    fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)
    # Middle
    fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)
    # Ring
    fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)
    # Pinky
    fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)

    return fingers

def smooth_move(prev_pos, current_pos, smooth_factor):
    """
    Smoothly interpolate between the last and current positions for smoother movement.
    """
    # If previous position is None, return current position directly
    if prev_pos is None:
        return current_pos
    x = int(prev_pos[0] + smooth_factor * (current_pos[0] - prev_pos[0]))
    y = int(prev_pos[1] + smooth_factor * (current_pos[1] - prev_pos[1]))
    return (x, y)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # Get important landmarks
        index_tip = lm[8]
        thumb_tip = lm[4]
        middle_tip = lm[12]
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

        # Map camera coordinates (which range 0-1) to full screen size with clamping
        sx = min(max(int(index_tip.x * screen_w), 0), screen_w - 1)
        sy = min(max(int(index_tip.y * screen_h), 0), screen_h - 1)


        # ------------------- Finger States -------------------
        finger_states = fingers_up(lm)

        # ------------------- Click Detection -------------------
        if finger_states[1] == 1 and finger_states[0] == 1:
            dist = distance(index_pos, (int(thumb_tip.x * w), int(thumb_tip.y * h)))
            if dist < 25 and time.time() - click_cooldown > 0.8:
                pyautogui.click()
                click_cooldown = time.time()
                cv2.putText(img, "Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ------------------- Scroll Detection -------------------
        if finger_states[1] == 1 and finger_states[2] == 1 and all(f == 0 for f in finger_states[3:]): 
            # Two fingers up (index & middle), rest down
            x_diff = abs(index_pos[0] - middle_pos[0])
            y_avg = (index_pos[1] + middle_pos[1]) // 2

            if x_diff < 30:
                if prev_scroll_y is not None:
                    delta = y_avg - prev_scroll_y
                    if abs(delta) > 10:
                        direction = "down" if delta > 0 else "up"
                        scroll_amount = 80 if direction == "down" else -80
                        pyautogui.scroll(scroll_amount)
                        cv2.putText(img, f"Scroll {direction}", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                prev_scroll_y = y_avg
            else:
                prev_scroll_y = None
        else:
            prev_scroll_y = None

        # ------------------- Mouse Movement (Annotation) -------------------
        if finger_states[1] == 1:  # Index finger up
            # Only smooth move if both prev_x and prev_y are valid
            if prev_x is not None and prev_y is not None:
                smooth_position = smooth_move((prev_x, prev_y), (sx, sy), smooth_factor)
                pyautogui.moveTo(smooth_position)
            else:
                pyautogui.moveTo(sx, sy)  # Initial movement, no smoothing yet

            prev_x, prev_y = sx, sy  # Update previous positions

        draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        # Hand lost - don't move the cursor anymore
        prev_x, prev_y = None, None

    cv2.imshow("Hand Mouse Controller", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
