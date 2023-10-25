import matplotlib.animation as animation
import RoadJunction as rj
import TrafficLightControlSystem as tlcs

DECREASE_CARS_RATE = 5
PARAMETERS_CHANGE_RATE = 250
EMERGENCY_THRESHOLD = 0.8


class Animator:
    def __init__(self, figure, function):
        self.animation = animation.FuncAnimation(figure, function, interval=75, cache_frame_data=False)
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

    def _update(self, i):
        if i == 0:
            self.__initialize_simulation()

        if i == self.marker:
            self.__update_marker_and_switch()

        if i % self.increase_cars_rate == 0:
            self.__adjust_car_number(self.plot.x_plot[0], 1)
            self.__adjust_car_number(self.plot.y_plot[0], 1)

        if i % DECREASE_CARS_RATE == 0:
            plot_to_decrease = self.plot.x_plot[0] if self.switch_x_y else self.plot.y_plot[0]
            self.__adjust_car_number(plot_to_decrease, -1)

        if i % PARAMETERS_CHANGE_RATE == 0:
            self.__update_air_transparency()
            self.__update_emergency()

        self.__update_legend(i)

    def __initialize_simulation(self):
        self.animator.animation.pause()
        self.plot.switch_lights(self.switch_x_y)
        self.marker = round(
            self.tlcs.perform_simulation(self.parameters.time_of_day,
                                         self.parameters.cars_queuing_x,
                                         self.parameters.air_transparency,
                                         self.parameters.emergency)
            * 10, 0)

    def __update_legend(self, i):
        legend_values = [i,
                         self.marker,
                         self.parameters.time_of_day,
                         self.plot.x_plot[0].get_linewidth(),
                         self.plot.y_plot[0].get_linewidth(),
                         self.parameters.air_transparency,
                         self.parameters.emergency]

        for idx, value in enumerate(legend_values):
            self.plot.update_legend_text(idx, value)

        if self.parameters.emergency > EMERGENCY_THRESHOLD:
            self.plot.legend.get_texts()[6].set_color('red')
        else:
            self.plot.legend.get_texts()[6].set_color('black')

    def __update_marker_and_switch(self):
        new_linewidth = self.plot.x_plot[0].get_linewidth() if self.switch_x_y else self.plot.y_plot[0].get_linewidth()

        self.marker += round(self.tlcs.perform_simulation(self.parameters.time_of_day,
                                                          new_linewidth,
                                                          self.parameters.air_transparency,
                                                          self.parameters.emergency)
                             * 10, 0)

        self.switch_x_y = not self.switch_x_y
        self.plot.switch_lights(self.switch_x_y)

        if self.parameters.emergency > EMERGENCY_THRESHOLD:
            self.parameters.change_emergency()

    def __adjust_car_number(self, plot, change):
        new_width = plot.get_linewidth() + change
        plot.set_linewidth(max(new_width, 0))

    def __update_air_transparency(self):
        self.parameters.change_air_transparency()
        self.plot.fig.patch.set_alpha((100 - self.parameters.air_transparency) / 100)

    def __update_emergency(self):
        self.parameters.change_emergency()
