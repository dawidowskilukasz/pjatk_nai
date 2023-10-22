import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

LEVELS = ['low', 'medium', 'high']

traffic_during_day = ctrl.Antecedent(np.arange(0, 24, 0.5), 'traffic_during_day')
cars_queuing = ctrl.Antecedent(np.arange(0, 15, 1), 'cars_queuing')
road_visibility = ctrl.Antecedent(np.arange(1, 100, 1), 'road_visibility')
emergency = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'emergency')

light_duration = ctrl.Consequent(np.arange(0, 10.5, 0.5), 'light_duration')

road_visibility.automf(3, names=LEVELS)

cars_queuing.automf(3, names=LEVELS)

x = fuzz.trapmf(traffic_during_day.universe, [0, 0, 4.5, 5.5])
y = fuzz.trapmf(traffic_during_day.universe, [21, 22, 23.5, 23.5])
z = [max(a, b) for a, b in zip(x, y)]
traffic_during_day['low'] = z

x = fuzz.trapmf(traffic_during_day.universe, [4.5, 5.5, 6.5, 7.5])
y = fuzz.trapmf(traffic_during_day.universe, [10, 11, 14, 15])
w = fuzz.trapmf(traffic_during_day.universe, [18, 19, 21, 22])
z = [max(a, b, c) for a, b, c in zip(x, y, w)]
traffic_during_day['medium'] = z

x = fuzz.trapmf(traffic_during_day.universe, [6.5, 7.5, 10, 11])
y = fuzz.trapmf(traffic_during_day.universe, [14, 15, 18, 19])
z = [max(a, b) for a, b in zip(x, y)]
traffic_during_day['high'] = z

x = np.ones(len(emergency.universe))
x[:-1] = 0.
y = np.zeros(len(emergency.universe))
y[:-1] = 1.
emergency['is_not'] = x
emergency['is'] = y

light_duration.automf(3, names=LEVELS)

cars_queuing.view()
road_visibility.view()
traffic_during_day.view()
emergency.view()
light_duration.view()

rule1 = ctrl.Rule(emergency['is'], light_duration['low'])
rule2 = ctrl.Rule(emergency['is_not'] & cars_queuing['high'], light_duration['high'])
rule3 = ctrl.Rule(traffic_during_day['high'] & cars_queuing['high'], light_duration['high'])
rule4 = ctrl.Rule(traffic_during_day['low'] | cars_queuing['low'], light_duration['low'])
rule5 = ctrl.Rule(traffic_during_day['low'] & cars_queuing['low'], light_duration['low'])
rule6 = ctrl.Rule(traffic_during_day['medium'] & cars_queuing['medium'], light_duration['medium'])
rule7 = ctrl.Rule(traffic_during_day['high'] & cars_queuing['low'], light_duration['medium'])
rule8 = ctrl.Rule(traffic_during_day['low'] & cars_queuing['high'], light_duration['medium'])
rule9 = ctrl.Rule(traffic_during_day['high'] & cars_queuing['high'] & road_visibility['high'], light_duration['high'])
rule10 = ctrl.Rule(traffic_during_day['low'] & cars_queuing['low'] & road_visibility['low'], light_duration['low'])
rule11 = ctrl.Rule(traffic_during_day['medium'] & cars_queuing['medium'] & road_visibility['medium'],
                   light_duration['medium'])
rule12 = ctrl.Rule(traffic_during_day['high'] & cars_queuing['high'] & road_visibility['low'], light_duration['medium'])
rule13 = ctrl.Rule(road_visibility['low'], light_duration['low'])
rule14 = ctrl.Rule(road_visibility['medium'], light_duration['medium'])
rule15 = ctrl.Rule(road_visibility['high'], light_duration['high'])
rule16 = ctrl.Rule(road_visibility['medium'] & cars_queuing['high'], light_duration['medium'])

traffic_lights_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11,
                                          rule12, rule13, rule14, rule15, rule16])


traffic_lights = ctrl.ControlSystemSimulation(traffic_lights_ctrl)

traffic_lights.input['emergency'] = 0
traffic_lights.input['traffic_during_day'] = 18
traffic_lights.input['cars_queuing'] = 15
traffic_lights.input['road_visibility'] = 10

traffic_lights.compute()

print(traffic_lights.output['light_duration'])
light_duration.view(sim=traffic_lights)

plt.show()
