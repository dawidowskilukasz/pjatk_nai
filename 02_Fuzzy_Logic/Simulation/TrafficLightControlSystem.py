import matplotlib.pyplot as plt
import numpy as np
import random

import skfuzzy as fuzz
from skfuzzy import control as ctrl

LEVELS = ['low', 'medium', 'high']
LIGHT_LEVELS = ['extremely low', 'very low', 'low', 'medium', 'high', 'very high', 'extremely high']

TIME_RANGE = np.array([0, 24])
CARS_RANGE = np.array([0, 40])
AIR_RANGE = np.array([0, 100])
EMERGENCY_RANGE = np.array([0, 1.1])
LIGHT_RANGE = np.array([0, 22])


def create_automf(ent):
    return ent.automf(3, names=LEVELS)


def combine_trapmf(ent, *value_sets):
    levels = [fuzz.trapmf(ent.universe, values) for values in value_sets]
    return [max(*values) for values in zip(*levels)]


class TrafficLightControlSystemSetup:
    def __init__(self):
        self.traffic_during_day = ctrl.Antecedent(np.arange(TIME_RANGE[0], TIME_RANGE[-1], 0.5), 'traffic_during_day')
        self.cars_queuing = ctrl.Antecedent(np.arange(CARS_RANGE[0], CARS_RANGE[-1], 1), 'cars_queuing')
        self.air_transparency = ctrl.Antecedent(np.arange(AIR_RANGE[0], AIR_RANGE[-1], 1), 'air_transparency')
        self.emergency = ctrl.Antecedent(np.arange(EMERGENCY_RANGE[0], EMERGENCY_RANGE[-1], 0.1), 'emergency')

        self.light_duration = ctrl.Consequent(np.arange(LIGHT_RANGE[0], LIGHT_RANGE[-1], 1), 'light_duration',
                                              'centroid')
        #   Controls which defuzzification method will be used.
        #   * 'centroid': Centroid of area
        #   * 'bisector': bisector of area
        #   * 'mom'     : mean of maximum
        #   * 'som'     : min of maximum
        #   * 'lom'     : max of maximum

        self.traffic_during_day['low'] = combine_trapmf(self.traffic_during_day,
                                                        [0, 0, 4.5, 5.5], [21, 22, 23.5, 23.5])
        self.traffic_during_day['medium'] = combine_trapmf(self.traffic_during_day,
                                                           [4.5, 5.5, 6.5, 7.5], [10, 11, 14, 15], [18, 19, 21, 22])
        self.traffic_during_day['high'] = combine_trapmf(self.traffic_during_day,
                                                         [6.5, 7.5, 10, 11], [14, 15, 18, 19])

        create_automf(self.cars_queuing)
        create_automf(self.air_transparency)

        self.emergency['is'] = np.concatenate([np.zeros(len(self.emergency.universe) - 2), np.ones(2)])
        self.emergency['is_not'] = np.concatenate([np.ones(len(self.emergency.universe) - 2), np.zeros(2)])

        self.light_duration.automf(7, names=LIGHT_LEVELS)

    def assess_time(self, value):
        time_assessment = [fuzz.interp_membership(self.traffic_during_day.universe,
                                                  self.traffic_during_day['high'].mf,
                                                  value),
                           fuzz.interp_membership(self.traffic_during_day.universe,
                                                  self.traffic_during_day['medium'].mf,
                                                  value),
                           fuzz.interp_membership(self.traffic_during_day.universe,
                                                  self.traffic_during_day['low'].mf,
                                                  value)]
        return time_assessment.index(max(time_assessment))


class TrafficLightControlSystemRules:
    def __init__(self):
        self.setup = TrafficLightControlSystemSetup()
        self.rules = self._create_rules()

    def _create_rules(self):
        rule_1 = ctrl.Rule((self.setup.traffic_during_day['low'] &
                            self.setup.cars_queuing['low'] &
                            self.setup.air_transparency['low']) |
                           self.setup.emergency['is'],
                           self.setup.light_duration['extremely low'])

        rule_2 = ctrl.Rule(((self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['low']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['low'])) &
                           self.setup.emergency['is_not'],
                           self.setup.light_duration['very low'])

        rule_3 = ctrl.Rule(((self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['low']) |
                            (self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['low'])) &
                           self.setup.emergency['is_not'],
                           self.setup.light_duration['low'])

        rule_4 = ctrl.Rule(((self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['low']) |
                            (self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['low'])) &
                           self.setup.emergency['is_not'],
                           self.setup.light_duration['medium'])

        rule_5 = ctrl.Rule(((self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['low']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['low'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['medium'])) &
                           self.setup.emergency['is_not'],
                           self.setup.light_duration['high'])

        rule_6 = ctrl.Rule(((self.setup.traffic_during_day['low'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['low']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['medium'] &
                             self.setup.air_transparency['high'])) &
                           self.setup.emergency['is_not'],
                           self.setup.light_duration['very high'])

        rule_7 = ctrl.Rule(((self.setup.traffic_during_day['medium'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['high']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['medium']) |
                            (self.setup.traffic_during_day['high'] &
                             self.setup.cars_queuing['high'] &
                             self.setup.air_transparency['high'])) &
                           self.setup.emergency['is_not'],
                           self.setup.light_duration['extremely high'])

        return [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7]


class TrafficLightControlSystem:
    def __init__(self):
        self.setup = TrafficLightControlSystemSetup()
        self.rules = TrafficLightControlSystemRules().rules
        self.control = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control)

    def perform_simulation(self, time_of_day, cars_queuing, air_transparency, emergency=0.):
        self.simulation.input['traffic_during_day'] = time_of_day
        self.simulation.input['cars_queuing'] = cars_queuing
        self.simulation.input['air_transparency'] = air_transparency
        self.simulation.input['emergency'] = emergency

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
        self.air_transparency = random.randint(AIR_RANGE.min(), AIR_RANGE.max())
        self.emergency = round(random.random(), 1)

    def get_random_parameters(self):
        return [self.time_of_day, self.air_transparency, self.cars_queuing_x, self.cars_queuing_y, self.emergency]

    def change_air_transparency(self):
        self.air_transparency = random.randint(0, 100)

    def change_emergency(self):
        self.emergency = round(random.random(), 1)
