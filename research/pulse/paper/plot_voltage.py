import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
os.chdir(os.path.dirname(__file__))


pwm1 = [2, 2, 0, 2, 0]
pwm2 = [0, 0, 2, 0, 2]


plt.figure(figsize=(16, 4))
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.step(range(len(pwm1)), pwm1, label='Electrode 1', color='blue')
plt.step(range(len(pwm2)), pwm2, label='Electrode 2', color='red', linestyle='dashed')
plt.xlabel('Index', fontsize=26)
plt.ylabel('Voltage [V]', fontsize=26)
plt.tick_params(labelsize=26)
plt.legend(fontsize=26, loc='upper right', bbox_to_anchor=(1, -0.1))

plt.savefig('../figures/voltage_wave.svg', bbox_inches='tight', pad_inches=0)

plt.show()
