import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

LEVELS = ['low', 'medium', 'high']

traffic_during_day = ctrl.Antecedent(np.arange(0, 24, 0.5), 'traffic_during_day')
cars_queuing = ctrl.Antecedent(np.arange(0, 15, 1), 'cars_queuing')
speed_of_arriving = ctrl.Antecedent(np.arange(1, 100, 1), 'speed_of_arriving')
emergency = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'emergency')

light_duration = ctrl.Consequent(np.arange(0, 10.5, 0.5), 'light_duration')

speed_of_arriving['low'] = fuzz.gaussmf(speed_of_arriving.universe, 0, 15)
speed_of_arriving['medium'] = fuzz.gaussmf(speed_of_arriving.universe, 50, 15)
speed_of_arriving['high'] = fuzz.gaussmf(speed_of_arriving.universe, 100, 15)

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

speed_of_arriving.view()
cars_queuing.view()
traffic_during_day.view()
emergency.view()
light_duration.view()

# 1. If there IS an *emergency*, then the *light_duration* will be LOW.
# 2. If the *traffic_during_day* is HIGH AND the number of *cars_queuing* is *HIGH*,
#    then the *light_duration* will be HIGH.
# 3. If the *speed_of_arriving* is HIGH, the *light_duration* will be LOW.

rule1 = ctrl.Rule(emergency['is_not'], light_duration['low'])
rule2 = ctrl.Rule(traffic_during_day['high'] | cars_queuing['high'], light_duration['high'])
rule3 = ctrl.Rule(speed_of_arriving['high'], light_duration['low'])

traffic_lights_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

traffic_lights = ctrl.ControlSystemSimulation(traffic_lights_ctrl)

traffic_lights.input['emergency'] = 0
traffic_lights.input['traffic_during_day'] = 12
traffic_lights.input['cars_queuing'] = 8
traffic_lights.input['speed_of_arriving'] = 50

traffic_lights.compute()

print(traffic_lights.output['light_duration'])
light_duration.view(sim=traffic_lights)

plt.show()
