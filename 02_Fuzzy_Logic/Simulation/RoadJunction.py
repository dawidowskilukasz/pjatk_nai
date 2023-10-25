import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LEGEND_LABELS = ['Frame: ', 'Next switch: ', 'Time of day: ', '\nX: ', 'Y: ', '\nAir transparency: ', 'Emergency: ']
AXIS_LIMS = ((1, 9), (1, 9))
ROAD_LINES_POSITIONS = [4, 6]
SCATTER_SIZE = 500
ROAD_LINES_STYLE = (0, (5, 10))


def update_lines(line_obj, x_data, y_data):
    line_obj.set_xdata(x_data)
    line_obj.set_ydata(y_data)


def update_scatters(scatter_obj, color):
    scatter_obj.set_color(color)
    scatter_obj.set_edgecolor('black')


class RoadJunction:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5), facecolor='grey')

        self.set_axis_and_figures(*AXIS_LIMS)

        self.draw_road_lines()

        self.x_plot = self.draw_lines(y=5)
        self.y_plot = self.draw_lines(x=5)

        self.x_scat = self.draw_scatters(4, 5)
        self.y_scat = self.draw_scatters(5, 4)

        self.legend = self.set_legend()

    def set_axis_and_figures(self, x_lims, y_lims):
        self.fig.patch.set_alpha(0)

        self.ax.axis('off')
        self.ax.set_xlim(x_lims)
        self.ax.set_ylim(y_lims)
        self.ax.set_box_aspect(1)

    def draw_lines(self, x=None, y=None, color='dimgrey', linestyle='solid'):
        x_values = [x] * 9 if x else list(range(1, 10))
        y_values = [y] * 9 if y else list(range(1, 10))
        return self.ax.plot(x_values, y_values, c=color, linestyle=linestyle, linewidth=1, solid_capstyle='butt',
                            zorder=1)

    def draw_scatters(self, x, y):
        return self.ax.scatter(x, y, c='white', edgecolor='black', marker='o', s=500, zorder=4)

    def draw_road_lines(self):
        for i in ROAD_LINES_POSITIONS:
            self.draw_lines(y=i, color='lightgrey', linestyle=ROAD_LINES_STYLE)
            self.draw_lines(x=i, color='lightgrey', linestyle=ROAD_LINES_STYLE)

    def set_legend(self):
        return plt.legend(
            [mpatches.Patch(visible=False) for _ in range(3)] +
            [self.x_plot[0], self.y_plot[0]] +
            [mpatches.Patch(visible=False) for _ in range(2)],
            LEGEND_LABELS, bbox_to_anchor=(1, 1),
            frameon=False)

    def update_legend_text(self, i, to_update):
        self.legend.get_texts()[i].set_text(LEGEND_LABELS[i] + str(to_update))

    def switch_lights(self, switch_to_x):
        if switch_to_x:
            update_scatters(self.x_scat, 'lime')
            update_scatters(self.y_scat, 'red')
            update_lines(self.x_plot[0], list(range(1, 10)), [5] * 9)
            update_lines(self.y_plot[0], [5] * 4, list(range(1, 5)))
        else:
            update_scatters(self.y_scat, 'lime')
            update_scatters(self.x_scat, 'red')
            update_lines(self.y_plot[0], [5] * 9, list(range(1, 10)))
            update_lines(self.x_plot[0], list(range(1, 5)), [5] * 4)
