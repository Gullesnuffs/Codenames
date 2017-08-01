from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import numpy as np

strings = [x.split(' ') for x in sys.stdin.read().strip().split('\n')]
print(strings)
values = [[float(x[0]), float(x[1])] for x in strings]

probabilities = []
step = 0.1
for x0 in np.arange(-0.5, 1.5, step):
    vs = [x for x in values if x[0] >= x0 and x[0] < x0 + step]
    if len(vs) > 0:
        prob = len([x for x in vs if x[1] == 1]) / len(vs)
        probabilities.append([x0, prob])


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['CMU Serif']
rcParams['font.size'] = 24
rcParams['mathtext.default'] = 'regular'

fig, ax = plt.subplots(figsize=(12, 12))

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']

# ax.plot([0.0, 1.0], [0.0, 1.0], color="#00000099", lw=2)
ax.scatter(x=[x[0] for x in probabilities], y=[x[1] for x in probabilities], color=colors[0], s=300)
plt.xlabel('Similarity')
plt.ylabel('Probability of choosing the word')

ax.scatter(x=[x[0] for x in values], y=[x[1] for x in values], color=colors[1], s=50, alpha=0.1)

pdf = PdfPages('out.pdf')
pdf.savefig(fig)
pdf.close()