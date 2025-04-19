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

        # Define ROI boundaries (tune these to your camera setup)
        roi_left = 0.2
        roi_right = 0.8
        roi_top = 0.2
        roi_bottom = 0.8

        # Normalize coordinates within ROI and map to screen
        norm_x = (index_tip.x - roi_left) / (roi_right - roi_left)
        norm_y = (index_tip.y - roi_top) / (roi_bottom - roi_top)

        # Clamp between 0 and 1
        norm_x = min(max(norm_x, 0.0), 1.0)
        norm_y = min(max(norm_y, 0.0), 1.0)

        padding = 5  # pixels away from edge to avoid fail-safe trigger
        sx = int(norm_x * (screen_w - 2 * padding)) + padding
        sy = int(norm_y * (screen_h - 2 * padding)) + padding




        # ------------------- Finger States -------------------
        finger_states = fingers_up(lm)

        # ------------------- Click Detection -------------------
# ------------------- Click Detection using L Shape -------------------
        if finger_states[1] == 1 and finger_states[0] == 1:  # Index and Thumb up
            wrist = lm[0]
            
            # Vectors from wrist to thumb tip and index tip
            v1 = (lm[4].x - wrist.x, lm[4].y - wrist.y)   # Thumb
            v2 = (lm[8].x - wrist.x, lm[8].y - wrist.y)   # Index

            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag_v1 = math.hypot(v1[0], v1[1])
            mag_v2 = math.hypot(v2[0], v2[1])

            if mag_v1 > 0 and mag_v2 > 0:
                angle_rad = math.acos(dot_product / (mag_v1 * mag_v2))
                angle_deg = math.degrees(angle_rad)

                # Show angle on screen for debugging
                cv2.putText(img, f"Angle: {int(angle_deg)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # If angle is around 90 degrees, trigger click
                if 35 <= angle_deg <= 65 and time.time() - click_cooldown > 0.8:
                    pyautogui.click()
                    click_cooldown = time.time()
                    cv2.putText(img, "L-Click", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Calculate angle between vectors using dot product


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
