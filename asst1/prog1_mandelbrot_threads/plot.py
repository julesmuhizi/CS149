import os
import numpy as np
import matplotlib.pyplot as plt

speedup_array= []
for index, threads in enumerate(np.arange(1,9)):
    print(threads)
    output = os.popen(f"./mandelbrot -view 2 -t {threads} | grep speedup").read()
    speedup = output.split()[0].split('x')[0].split('(')[-1]
    print(f'threads: {threads} - speedup {speedup}')
    speedup_array.append(float(speedup))
plt.plot(np.arange(1,9),speedup_array)
plt.xlabel('# threads')
plt.ylabel('speedup over 1 thread')
plt.title('# of threads vs speedup')
plt.savefig('plot.png')
plt.show()

