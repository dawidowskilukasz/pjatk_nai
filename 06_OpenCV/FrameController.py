import time

INTERVAL = 250000000


class FrameController:
    """
    Controls the timing and intervals between frames.
    """
    def __init__(self):
        """
        Initialize the FrameController with initial timestamp and flags.
        """
        self.prev_frame = time.time_ns()
        self.next_frame = None
        self.first_frame_for_click = True
        self.first_frame_for_tab = True

    def set_next_frame(self):
        """
        Set the timestamp for the next frame.
        """
        self.next_frame = time.time_ns()

    def set_prev_frame(self):
        """
        Set the timestamp for the previous frame.
        """
        self.prev_frame = self.next_frame

    def set_first_frame_for_click(self):
        """
        Set the flag indicating it's not the first frame for a click action.
        """
        self.first_frame_for_click = False

    def set_first_frame_for_tab(self):
        """
        Set the flag indicating it's not the first frame for a tab action.
        """
        self.first_frame_for_tab = False

    def is_interval_elapsed(self):
        """
        Check if the interval between frames has elapsed.
        """
        return self.next_frame - self.prev_frame > INTERVAL

    def get_next_frame(self):
        """
        Get the timestamp of the next frame.
        """
        return self.next_frame

    def is_first_frame_for_click(self):
        """
        Check if it's the first frame for a click action.
        """
        return self.first_frame_for_click

    def is_first_frame_for_tab(self):
        """
        Check if it's the first frame for a tab action.
        """
        return self.first_frame_for_tab
