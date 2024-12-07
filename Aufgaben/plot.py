import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.use('TkAgg')

x = []
y = []

for i in range(10):
    x.append(i + random.randint(0, 3))
    y.append(i + random.randint(0, 3))

plt.scatter(x, y)
plt.show()