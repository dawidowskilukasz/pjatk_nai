import cv2
import mediapipe as mp
import pyautogui
import time
import ctypes


def get_finger_coordinate(finger, image_shape):
    return image_shape - (finger * image_shape)


def compare_coordinates(finger_1, finger_2, image_shape):
    return get_finger_coordinate(finger_1, image_shape) - get_finger_coordinate(finger_2, image_shape)


def compare_group_of_coordinates(coordinates_array, image_shape):
    comparison_result = True
    for i in range(0, len(coordinates_array), 2):
        if compare_coordinates(coordinates_array[i], coordinates_array[i + 1], image_shape) > 0:
            comparison_result = False
            break
    return comparison_result


screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

prev_frame = time.time_ns()
first_frame_for_click = True
first_frame_for_tab = True

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
                # TO DO
                # 1. Save last frame to calculate position of finger and base cursor movement on it

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_finger_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                middle_finger_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                ring_finger_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                pinky_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

                next_frame = time.time_ns()

                if abs(compare_coordinates(index_finger_MCP.x, pinky_MCP.x, image.shape[1])) < 80:

                    if compare_group_of_coordinates([index_finger_tip.x, index_finger_dip.x,
                                                     middle_finger_tip.x, middle_finger_dip.x,
                                                     ring_finger_tip.x, ring_finger_dip.x,
                                                     pinky_tip.x, pinky_dip.x], image.shape[1]):

                        if compare_group_of_coordinates([index_finger_tip.y, index_finger_dip.y,
                                                         middle_finger_tip.y, middle_finger_dip.y,
                                                         ring_finger_tip.y, ring_finger_dip.y,
                                                         pinky_tip.y, pinky_dip.y], image.shape[0]):

                            pyautogui.scroll(-100)

                        else:

                            pyautogui.scroll(100)

                    else:
                        if first_frame_for_click:
                            prev_coordinate_click = get_finger_coordinate(index_finger_tip.y, image.shape[0])
                            first_frame_for_click = False

                        next_coordinate_click = get_finger_coordinate(index_finger_tip.y, image.shape[0])
                        if next_frame - prev_frame > 250000000:
                            if next_coordinate_click - prev_coordinate_click > 30:
                                pyautogui.leftClick()
                            prev_coordinate_click = next_coordinate_click
                            prev_frame = next_frame

                elif compare_group_of_coordinates([middle_finger_tip.y, middle_finger_MCP.y,
                                                     ring_finger_tip.y, ring_finger_MCP.y,
                                                     pinky_tip.y, pinky_MCP.y], image.shape[0]):

                    target_x = get_finger_coordinate(index_finger_tip.x, image.shape[1])
                    target_y = get_finger_coordinate(index_finger_tip.y, image.shape[0])

                    target_x = target_x / image.shape[1] * screen_width
                    target_y = (image.shape[0] - target_y) / image.shape[0] * screen_height

                    current_x, current_y = pyautogui.position()

                    move_x = target_x - current_x
                    move_y = target_y - current_y

                    pyautogui.move(move_x, move_y)

                else:
                    if first_frame_for_tab:
                        prev_coordinate_tab = get_finger_coordinate(index_finger_tip.y, image.shape[0])
                        first_frame_for_tab = False

                    next_coordinate_tab = get_finger_coordinate(thumb_tip.x, image.shape[1])

                    if next_frame - prev_frame > 250000000:
                        if next_coordinate_tab - prev_coordinate_tab > 50:
                            pyautogui.hotkey('ctrl', 'tab')
                        prev_coordinate_tab = next_coordinate_tab
                        prev_frame = next_frame


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
