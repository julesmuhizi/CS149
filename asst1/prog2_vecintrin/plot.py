import matplotlib.pyplot as plt
import os
import numpy as np

utilization = [77.4, 70.3, 66.6,64.9]
nums = [2,4,8,16]
plt.plot(nums, utilization)
plt.title("Utilization vs vector width")
plt.xlabel("Vector Width")
plt.ylabel("Utilization")
plt.savefig("prog2_util.png")

