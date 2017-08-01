from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import numpy as np
import matplotlib.cm as cm

strings = [x.split(' ') for x in sys.stdin.read().strip().split('\n')]
# print(strings)
values = [[float(x[0]), float(x[1]), float(x[2])] for x in strings]

step = 0.1
steps = int(1.01 / step)
probabilities = np.zeros((steps, steps))
for x0i in range(0,steps):
    x0 = x0i * step
    for y0i in range(0,steps):
        y0 = y0i * step
        vs = [x for x in values if x[0] >= x0 and x[0] < x0 + step and x[1] >= y0 and x[1] < y0 + step]

        if len(vs) > 0:
            print(vs)
            prob = len([x for x in vs if x[2] > 0.5]) / len(vs)
        else:
            prob = 0

        probabilities[x0i,y0i] = prob


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 24
rcParams['mathtext.default'] = 'regular'

fig, ax = plt.subplots(figsize=(12, 12))

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']

plt.imshow(probabilities, extent=(0.0, 1.0, 0.0, 1.0), cmap=cm.seismic, origin='lower')
plt.colorbar()

pdf = PdfPages('out.pdf')
pdf.savefig(fig)
pdf.close()