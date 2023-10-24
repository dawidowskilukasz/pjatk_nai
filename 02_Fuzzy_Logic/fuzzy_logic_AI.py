'''
This is a model of traffic light control system which uses fuzzy logic.

This model shows how would traffic lights work based on few input variables like traffic density based on time of day,
car queuing at the intersection, air transparency or in other words road visibility (rain, fog etc.) and the presence
of an emergency situation

In the code we can find variables, membership functions, rules to define fuzzy logic which allow it to simulate the
control system of traffic lights to determine optimal green light duration

Authors:
By Maciej Zagórski (s23575) and Łukasz Dawidowski (s22621), group 72c (10:15-11:45)

Usage:
- Modify the input values for 'emergency,' 'traffic_during_day,' 'cars_queuing,' and 'air_transparency.'
- Run the code to compute the optimal 'light_duration.'
- Visualize the results using the plotted membership functions.
'''
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

LEVELS = ['low', 'medium', 'high']

traffic_during_day = ctrl.Antecedent(np.arange(0, 24, 0.5), 'traffic_during_day')
cars_queuing = ctrl.Antecedent(np.arange(0, 15, 1), 'cars_queuing')
air_transparency = ctrl.Antecedent(np.arange(1, 100, 1), 'air_transparency')
emergency = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'emergency')

light_duration = ctrl.Consequent(np.arange(0, 10.5, 0.5), 'light_duration')

air_transparency.automf(3, names=LEVELS)

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
air_transparency.view()
traffic_during_day.view()
emergency.view()
light_duration.view()

rule_1 = ctrl.Rule((traffic_during_day['high'] & cars_queuing['high']) |
                   (air_transparency['high'])
                   , light_duration['high'])
rule_2 = ctrl.Rule((traffic_during_day['medium'] & cars_queuing['medium']) |
                   (traffic_during_day['high'] & cars_queuing['low']) |
                   (traffic_during_day['low'] & cars_queuing['high']) |
                   # (air_transparency['medium']) |
                   (air_transparency['medium'] & cars_queuing['high']) |
                   (traffic_during_day['high'] & cars_queuing['high'] &
                    air_transparency['low'])
                   , light_duration['medium'])
rule_3 = ctrl.Rule((traffic_during_day['low'] & cars_queuing['low']) |
                   (air_transparency['low'])
                   , light_duration['low'])

traffic_lights_ctrl = ctrl.ControlSystem([rule_1, rule_2, rule_3])

traffic_lights = ctrl.ControlSystemSimulation(traffic_lights_ctrl)

# traffic_lights.input['emergency'] = 1
traffic_lights.input['traffic_during_day'] = 15
traffic_lights.input['cars_queuing'] = 7
traffic_lights.input['air_transparency'] = 15

traffic_lights.compute()

print(traffic_lights.output['light_duration'])
light_duration.view(sim=traffic_lights)

plt.show()
