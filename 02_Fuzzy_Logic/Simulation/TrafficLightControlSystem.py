import matplotlib.pyplot as plt
import numpy as np
import random

import skfuzzy as fuzz
from skfuzzy import control as ctrl

LEVELS = ['low', 'medium', 'high']
TIME_RANGE = np.array([0, 24])
CARS_RANGE = np.array([0, 40])


def normalize(value, normalization_range):
    return round((value - normalization_range.min()) / (normalization_range.max() - normalization_range.min()) * 100)


def normalize_list(values):
    return [normalize(value, TIME_RANGE) for value in values]


def create_antecedent(name):
    return ctrl.Antecedent(np.arange(0, 100, 1), name)


def create_automf(ent):
    return ent.automf(3, names=LEVELS)


def create_trapmf(ent, values):
    return fuzz.trapmf(ent.universe, normalize_list(values))


def create_trap_combmf(ent, *value_sets):
    levels = [create_trapmf(ent, values) for values in value_sets]
    return [max(*values) for values in zip(*levels)]


class TrafficLightControlSystemSetup:
    def __init__(self):
        self.traffic_during_day = create_antecedent('traffic_during_day')
        self.cars_queuing = create_antecedent('cars_queuing')
        self.air_transparency = create_antecedent('air_transparency')
        self.emergency = create_antecedent('emergency')

        self.light_duration = ctrl.Consequent(np.arange(0, 15, 1), 'light_duration', 'centroid')
        #   Controls which defuzzification method will be used.
        #   * 'centroid': Centroid of area
        #   * 'bisector': bisector of area
        #   * 'mom'     : mean of maximum
        #   * 'som'     : min of maximum
        #   * 'lom'     : max of maximum

        self.traffic_during_day['low'] = create_trap_combmf(self.traffic_during_day,
                                                            [0, 0, 4.5, 5.5], [21, 22, 23.5, 23.5])
        self.traffic_during_day['medium'] = create_trap_combmf(self.traffic_during_day,
                                                               [4.5, 5.5, 6.5, 7.5], [10, 11, 14, 15], [18, 19, 21, 22])
        self.traffic_during_day['high'] = create_trap_combmf(self.traffic_during_day,
                                                             [6.5, 7.5, 10, 11], [14, 15, 18, 19])

        create_automf(self.cars_queuing)
        create_automf(self.air_transparency)

        self.emergency['is_not'] = np.concatenate([np.zeros(len(self.emergency.universe) - 10), np.ones(10)])
        self.emergency['is'] = np.concatenate([np.ones(len(self.emergency.universe) - 10), np.zeros(10)])

        create_automf(self.light_duration)

    def assess_time(self, value):
        time_assessment = [fuzz.interp_membership(self.traffic_during_day.universe, self.traffic_during_day['high'].mf,
                                                  normalize(value, TIME_RANGE)),
                           fuzz.interp_membership(self.traffic_during_day.universe, self.traffic_during_day['high'].mf,
                                                  normalize(value, TIME_RANGE)),
                           fuzz.interp_membership(self.traffic_during_day.universe, self.traffic_during_day['high'].mf,
                                                  normalize(value, TIME_RANGE))]
        return time_assessment.index(max(time_assessment))


class TrafficLightControlSystem:
    def __init__(self):
        self.setup = TrafficLightControlSystemSetup()
        self.rules = self._create_rules()
        self.control = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control)

    def _create_rules(self):
        rule_1 = ctrl.Rule((self.setup.traffic_during_day['high'] & self.setup.cars_queuing['high']) |
                           (self.setup.air_transparency['high'])
                           , self.setup.light_duration['high'])
        rule_2 = ctrl.Rule((self.setup.traffic_during_day['medium'] & self.setup.cars_queuing['medium']) |
                           (self.setup.traffic_during_day['high'] & self.setup.cars_queuing['low']) |
                           (self.setup.traffic_during_day['low'] & self.setup.cars_queuing['high']) |
                           # (self.setup.air_transparency['medium']) |
                           (self.setup.air_transparency['medium'] & self.setup.cars_queuing['high']) |
                           (self.setup.traffic_during_day['high'] & self.setup.cars_queuing['high'] &
                            self.setup.air_transparency['low'])
                           , self.setup.light_duration['medium'])
        rule_3 = ctrl.Rule((self.setup.traffic_during_day['low'] & self.setup.cars_queuing['low']) |
                           (self.setup.air_transparency['low'])
                           , self.setup.light_duration['low'])
        return [rule_1, rule_2, rule_3]

    def perform_simulation(self, time_of_day, cars_queuing, air_transparency, emergency=0):
        self.simulation.input['traffic_during_day'] = normalize(time_of_day, TIME_RANGE)
        self.simulation.input['cars_queuing'] = normalize(cars_queuing, CARS_RANGE)
        self.simulation.input['air_transparency'] = air_transparency
        # self.simulation.input['emergency'] = emergency

        self.simulation.compute()

        return self.simulation.output['light_duration']

    def show_views(self):
        self.setup.traffic_during_day.view()
        self.setup.cars_queuing.view()
        self.setup.air_transparency.view()
        self.setup.emergency.view()
        self.setup.light_duration.view(sim=self.simulation)
        plt.show()


class RandomParameters:

    def __init__(self):
        self.time_of_day = random.randint(TIME_RANGE.min(), TIME_RANGE.max() * 2) / 2
        self.cars_queuing_x = random.randint(CARS_RANGE.min(), CARS_RANGE.max())
        self.cars_queuing_y = random.randint(CARS_RANGE.min(), CARS_RANGE.max())
        self.air_transparency = random.randint(0, 100)
        self.emergency = random.randint(0, 100)

    def get_random_parameters(self):
        return [self.time_of_day, self.air_transparency, self.cars_queuing_x, self.cars_queuing_y, self.emergency]

    def change_air_transparency(self):
        self.air_transparency = random.randint(0, 100)
