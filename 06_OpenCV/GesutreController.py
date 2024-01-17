import cv2
import mediapipe as mp
import pyautogui
import ctypes
import FrameController
import FingersController as fc
import Actions

MOVE_SCROLL_THRESHOLD = 60
TAB_CHANGE_THRESHOLD = 40
CLICK_THRESHOLD = 20

frame_controller = FrameController.FrameController()

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

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

                fingers = fc.Fingers(hand_landmarks, mp_hands)

                frame_controller.set_next_frame()

                if fc.compare_fingers(fingers.index["MCP"].x, fingers.pinky["MCP"].x, image.shape[1],
                                      MOVE_SCROLL_THRESHOLD, True):

                    if fc.compare_group_of_fingers(fingers, image.shape[1], "x"):

                        Actions.scroll_window(fingers, image.shape[0])

                    else:

                        if frame_controller.is_first_frame_for_tab():
                            fingers_controller_tab = fc.FingersController(fingers.thumb['tip'].x, image.shape[1])
                            frame_controller.set_first_frame_for_tab()

                        fingers_controller_tab.set_next_coordinate(fingers.thumb['tip'].x)

                        if frame_controller.is_interval_elapsed():

                            if fingers_controller_tab.are_coordinates_changed(TAB_CHANGE_THRESHOLD):
                                pyautogui.hotkey('ctrl', 'tab')

                            fingers_controller_tab.set_prev_coordinates()
                            frame_controller.set_prev_frame()

                else:

                    if not (fc.compare_fingers(fingers.index["tip"].y, fingers.thumb["tip"].y, image.shape[0])):

                        Actions.move_mouse(fingers.index["tip"], image.shape)

                    else:

                        if frame_controller.is_first_frame_for_click():
                            fingers_controller_click = fc.FingersController(fingers.thumb['tip'].x, image.shape[1])
                            frame_controller.set_first_frame_for_click()

                        fingers_controller_click.set_next_coordinate(fingers.thumb['tip'].x)

                        if frame_controller.is_interval_elapsed():

                            if fingers_controller_click.are_coordinates_changed(CLICK_THRESHOLD):
                                pyautogui.leftClick()

                            fingers_controller_click.set_prev_coordinates()
                            frame_controller.set_prev_frame()

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
