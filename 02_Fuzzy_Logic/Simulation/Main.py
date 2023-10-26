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
from Animation import Animation
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = Animation()
    plt.show()
