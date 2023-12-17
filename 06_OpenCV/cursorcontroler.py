import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                #TO DO
                # 1. Save last frame to calculate position of finger and base cursor movement on it

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                lmbgest1_x, lmbgest1_y = (
                    int(index_finger_MCP.x * image.shape[1]),
                    int(index_finger_MCP.y * image.shape[0])
                )
                lmbgest2_x, lmbgest2_y = (
                    int(thumb_tip.x * image.shape[1]),
                    int(thumb_tip.y * image.shape[0])
                )

                if lmbgest1_y > lmbgest2_y:
                    pyautogui.leftClick()

                target_x, target_y = (
                    int(index_finger_tip.x * image.shape[1]),
                    int(index_finger_tip.y * image.shape[0])
                )
                print(f"Index finger tip position x: {target_x} Y: {target_y}")

                current_x, current_y = pyautogui.position()
                print(f"Current mouse position: ({current_x}, {current_y})")

                move_x = target_x - current_x
                move_y = target_y - current_y

                pyautogui.move(move_x, move_y)

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('Mouse Controler', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
