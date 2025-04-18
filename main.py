import cv2
import numpy as np
import pyautogui
from gesture_module import HandDetector

# Screen size
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Smoothening variables
prev_x, prev_y = 0, 0
smoothening = 5

click_threshold = 30  # Distance threshold for click gesture
clicked = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # Get coordinates of index finger tip
        x1, y1 = lmList[8][1], lmList[8][2]

        # Convert coordinates
        screen_x = np.interp(x1, [0, 640], [0, screen_width])
        screen_y = np.interp(y1, [0, 480], [0, screen_height])

        # Smooth movement
        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Gesture: Left click (pinch index and middle finger)
        length, _ = detector.findDistance(8, 12, img, draw=True)

        if length < click_threshold:
            if not clicked:
                pyautogui.click()
                clicked = True
        else:
            clicked = False

        # Draw pointer on webcam feed
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

    # Display the image
    cv2.imshow("Virtual Mouse", img)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
