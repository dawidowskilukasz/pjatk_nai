import ctypes


def get_finger_coordinate(finger, image_shape):
    return image_shape - (finger * image_shape)


def convert_finger_coordinates_to_screen(finger, image):
    x = get_finger_coordinate(finger.x, image[1]) / image[1] * ctypes.windll.user32.GetSystemMetrics(0)
    y = (image[0] - get_finger_coordinate(finger.y, image[0])) / image[0] * ctypes.windll.user32.GetSystemMetrics(1)
    return x, y


def subtract_fingers(finger_a, finger_b, image_shape):
    return get_finger_coordinate(finger_a, image_shape) - get_finger_coordinate(finger_b, image_shape)


def compare_fingers(finger_a, finger_b, image_shape, threshold=0, absolute=False):
    if absolute:
        return abs(subtract_fingers(finger_a, finger_b, image_shape)) < threshold
    else:
        return subtract_fingers(finger_a, finger_b, image_shape) < threshold


def compare_group_of_fingers(fingers, image_shape, axis, threshold=0):
    for finger in fingers.fingers:
        if not (compare_fingers(finger["tip"].x if axis == "x" else finger["tip"].y,
                                finger["dip"].x if axis == "x" else finger["dip"].y, image_shape, threshold)):
            return False
    return True


class Fingers:
    def __init__(self, hand_landmarks, mp_hands):
        self.hand_landmark = hand_landmarks.landmark
        self.mp_hand_landmark = mp_hands.HandLandmark

        self.index = {
            "tip": self.get_landmark("INDEX_FINGER_TIP"),
            "dip": self.get_landmark("INDEX_FINGER_DIP"),
            "pip": self.get_landmark("INDEX_FINGER_PIP"),
            "MCP": self.get_landmark("INDEX_FINGER_MCP")
        }
        self.middle = {
            "tip": self.get_landmark("MIDDLE_FINGER_TIP"),
            "dip": self.get_landmark("MIDDLE_FINGER_DIP"),
            "pip": self.get_landmark("MIDDLE_FINGER_PIP"),
            "MCP": self.get_landmark("MIDDLE_FINGER_MCP"),
        }
        self.ring = {
            "tip": self.get_landmark("RING_FINGER_TIP"),
            "dip": self.get_landmark("RING_FINGER_DIP"),
            "pip": self.get_landmark("RING_FINGER_PIP"),
            "MCP": self.get_landmark("RING_FINGER_MCP"),
        }
        self.pinky = {
            "tip": self.get_landmark("PINKY_TIP"),
            "dip": self.get_landmark("PINKY_DIP"),
            "pip": self.get_landmark("PINKY_PIP"),
            "MCP": self.get_landmark("PINKY_MCP"),
        }

        self.fingers = [
            self.index,
            self.middle,
            self.ring,
            self.pinky
        ]

        self.thumb = {
            "tip": self.get_landmark("THUMB_TIP"),
            "MCP": self.get_landmark("THUMB_MCP")
        }

    def get_landmark(self, landmark):
        return self.hand_landmark[self.mp_hand_landmark[landmark]]


class FingersController:
    def __init__(self, finger, image_shape):
        self.image_shape = image_shape
        self.prev_coordinate = get_finger_coordinate(finger, self.image_shape)
        self.next_coordinate = None

    def set_next_coordinate(self, finger):
        self.next_coordinate = get_finger_coordinate(finger, self.image_shape)

    def set_prev_coordinates(self):
        self.prev_coordinate = self.next_coordinate

    def are_coordinates_changed(self, threshold):
        return self.next_coordinate - self.prev_coordinate > threshold
