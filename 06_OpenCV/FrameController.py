import time

INTERVAL = 250000000


class FrameController:
    def __init__(self):
        self.prev_frame = time.time_ns()
        self.next_frame = None
        self.first_frame_for_click = True
        self.first_frame_for_tab = True

    def set_next_frame(self):
        self.next_frame = time.time_ns()

    def set_prev_frame(self):
        self.prev_frame = self.next_frame

    def set_first_frame_for_click(self):
        self.first_frame_for_click = False

    def set_first_frame_for_tab(self):
        self.first_frame_for_tab = False

    def is_interval_elapsed(self):
        return self.next_frame - self.prev_frame > INTERVAL

    def get_next_frame(self):
        return self.next_frame

    def is_first_frame_for_click(self):
        return self.first_frame_for_click

    def is_first_frame_for_tab(self):
        return self.first_frame_for_tab
