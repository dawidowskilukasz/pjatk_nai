import matplotlib.animation as animation
import RoadJunction as rj
import TrafficLightControlSystem as tlcs

LEGEND_PARAMETERS_INDEX = 2
DECREASE_CARS_RATE = 5
AIR_CHANGE_RATE = 500

class Animator:
    def __init__(self, figure, function):
        self.animation = animation.FuncAnimation(figure, function, interval=100, cache_frame_data=False)
        self.paused = True
        figure.canvas.mpl_connect('button_press_event', self._toggle_pause)

    def _toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused


class Animation:
    def __init__(self):
        self.marker = 0
        self.tlcs = tlcs.TrafficLightControlSystem()
        self.parameters = tlcs.RandomParameters()
        self.increase_cars_rate = (self.tlcs.setup.assess_time(self.parameters.time_of_day) + 1) * 10

        self.plot = rj.RoadJunction()
        self._initialize_plot()

        self.switch_x_y = True

        self.animator = Animator(self.plot.fig, self._update)

    def _initialize_plot(self):
        self.plot.fig.patch.set_alpha((100 - self.parameters.air_transparency) / 100)

        self.plot.x_plot[0].set_linewidth(self.parameters.cars_queuing_x)
        self.plot.y_plot[0].set_linewidth(self.parameters.cars_queuing_y)

        [self.plot.update_legend_text(i, self.parameters.get_random_parameters()[i - LEGEND_PARAMETERS_INDEX]) for i in
         range(LEGEND_PARAMETERS_INDEX, len(self.plot.legend.get_texts()))]

    def _update(self, i):
        if i == 0:
            self.__initialize_simulation()

        if i == self.marker:
            self.__update_marker_and_switch()

        if i % self.increase_cars_rate == 0:
            self.__increase_cars_number()

        if i % DECREASE_CARS_RATE == 0:
            self.__decrease_cars_number()

        if i % AIR_CHANGE_RATE == 0:
            self.__update_air_transparency()

        self.__update_legend(i)

    def __initialize_simulation(self):
        self.animator.animation.pause()
        self.plot.switch_lights(self.switch_x_y)
        self.marker = round(
            self.tlcs.perform_simulation(self.parameters.time_of_day, self.parameters.cars_queuing_x,
                                         self.parameters.air_transparency) * 10, 0)

    def __update_legend(self, i):
        self.plot.update_legend_text(0, i)
        self.plot.update_legend_text(1, self.marker)
        self.plot.update_legend_text(3, self.parameters.air_transparency)
        self.plot.update_legend_text(4, self.plot.x_plot[0].get_linewidth())
        self.plot.update_legend_text(5, self.plot.y_plot[0].get_linewidth())

    def __update_marker_and_switch(self):
        new_linewidth = self.plot.x_plot[0].get_linewidth() if self.switch_x_y else self.plot.y_plot[0].get_linewidth()
        self.marker += round(self.tlcs.perform_simulation(self.parameters.time_of_day, new_linewidth,
                                                          self.parameters.air_transparency) * 10, 0)
        self.switch_x_y = not self.switch_x_y
        self.plot.switch_lights(self.switch_x_y)

    def __increase_cars_number(self):
        self.plot.x_plot[0].set_linewidth((self.plot.x_plot[0].get_linewidth()) + 1)
        self.plot.y_plot[0].set_linewidth((self.plot.y_plot[0].get_linewidth()) + 1)

    def __decrease_cars_number(self):
        plot_do_decrease = self.plot.x_plot[0] if self.switch_x_y else self.plot.y_plot[0]
        if plot_do_decrease.get_linewidth() > 0:
            plot_do_decrease.set_linewidth(plot_do_decrease.get_linewidth() - 1)

    def __update_air_transparency(self):
        self.parameters.change_air_transparency()
        self.plot.fig.patch.set_alpha((100 - self.parameters.air_transparency) / 100)
