import pyautogui
import FingersController as fc


def scroll_window(fingers, image_shape):
    if fc.compare_group_of_fingers(fingers, image_shape, "y"):
        pyautogui.scroll(-100)
    else:
        pyautogui.scroll(100)


def move_mouse(finger, image_shape):
    target_x, target_y = fc.convert_finger_coordinates_to_screen(finger, image_shape)
    current_x, current_y = pyautogui.position()
    move_x, move_y = target_x - current_x, target_y - current_y
    pyautogui.move(move_x, move_y)
