import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation


# https://matplotlib.org/stable/api/animation_api.html
# https://matplotlib.org/stable/gallery/animation/pause_resume.html
# https://matplotlib.org/stable/gallery/animation/rain.html

class Animation:
    def __init__(self):

        n = [i for i in range(1, 9)]

        fig, ax_horiz = plt.subplots()

        ax_horiz.axis('off')

        ax_horiz.set_xlim(1, 8)
        ax_horiz.set_ylim(1, 8)

        ax_horiz_lights = ax_horiz
        ax_verti = ax_horiz
        ax_verti_lights = ax_horiz

        self.horiz_plot = ax_horiz.plot(n, np.array([[4, 5]] * len(n)), c='lightgrey', zorder=1, linewidth=1)
        self.horiz_scat = ax_horiz_lights.scatter([6, 3], [5, 4], c='white', edgecolor='black', marker='o', s=100,
                                                  zorder=2)

        self.verti_plot = ax_verti.plot(np.array([[4, 5]] * len(n)), n, c='lightgrey', zorder=1, linewidth=1)
        self.verti_scat = ax_verti_lights.scatter([4, 5], [6, 3], c='white', edgecolor='black', marker='o', s=100,
                                                  zorder=2)

        self.legend = plt.legend(self.horiz_plot + self.verti_plot,
                                 [self.horiz_plot[0].get_linewidth(), 'line B', 'line C', 'line D'],
                                 loc='upper right', frameon=False)

        self.animation = animation.FuncAnimation(fig, self.update, interval=100, cache_frame_data=False)
        self.paused = True

        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, i):
        if i == 0:
            self.animation.pause()
        else:
            for hor_plt in self.horiz_plot:
                hor_plt.set_linewidth(i / 2)
            self.legend.get_texts()[0].set_text(self.horiz_plot[0].get_linewidth())
        return self.horiz_plot, self.horiz_scat, self.verti_plot, self.verti_scat, self.legend


a = Animation()
plt.show()
