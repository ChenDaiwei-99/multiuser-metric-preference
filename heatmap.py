import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

fname = 'color_experiment_N48_165264247294_Mcrowd.mat'
key = 'M_all_mean'
M = loadmat(fname)[key]

matplotlib.rcParams['pdf.fonttype'] = 42    
matplotlib.rcParams['ps.fonttype'] = 42    
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(1,1)
img = ax.imshow(M)
labels = ['Lightness', 'Red vs.\nGreen', 'Blue vs.\n Yellow']
ax.set_xticks([0. , 1., 2.])
ax.set_xticklabels(labels, rotation=30)
ax.set_yticks([0. , 1., 2.])
ax.set_yticklabels(labels, rotation=30)
fig.colorbar(img)
plt.savefig('heatmap.pdf', bbox_inches='tight')
plt.show()