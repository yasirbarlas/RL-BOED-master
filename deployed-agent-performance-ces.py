from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

L = 1e7
T = 10

x = np.linspace(1, T, T)

redq = [4.2492,  7.9563, 10.9724, 12.5223, 13.0118, 13.2693, 13.4418, 13.5781,
        13.6925, 13.7818]

sbr = [4.3216,  8.0156, 10.9625, 12.3914, 12.8534, 13.1150, 13.2874, 13.4291,
       13.5442, 13.6402]

droq = [4.0685,  7.9073, 10.8860, 12.6041, 13.2825, 13.6247, 13.8031, 13.9199,
        14.0149, 14.0898]

sunrise = [4.1949,  7.9303, 10.7114, 12.2188, 12.7885, 13.0714, 13.2587, 13.4082,
           13.5273, 13.6330]

random = [3.3904,  5.8664,  7.6979,  8.6997,  9.3661,  9.8701, 10.2817, 10.6145,
          10.8940, 11.1452]

fig, ax = plt.subplots(figsize=(8, 4.8))
#fig, ax = plt.subplots()
#ax.hlines(np.log(L + 1), xmin = 0, xmax = T, color="black", linestyles="dashed")
ax.plot(x, redq, label="REDQ")
ax.plot(x, sbr, label="SBR")
ax.plot(x, droq, label="DroQ")
ax.plot(x, sunrise, label="SUNRISE")
ax.plot(x, random, label="Random")
ax.set_xlim(0, T)
ax.set_xlabel("Experiment")
ax.set_ylabel("sPCE")
ax.set_title("Average sPCE for Deployed Agents")
ax.grid()
ax.legend()

inset_position = [0.5, 0.05, 2.6, 1.4]  # [x0, y0, width, height]
ax_inset = inset_axes(ax, width=inset_position[2], height=inset_position[3], loc='lower left',
                      bbox_to_anchor=(inset_position[0], inset_position[1], 
                                      inset_position[2], inset_position[3]), 
                      bbox_transform=ax.transAxes)
#ax_inset.hlines(np.log(L + 1), xmin = 0, xmax = T, color="black", linestyles="dashed")
ax_inset.plot(x, redq, label="REDQ")
ax_inset.plot(x, sbr, label="SBR")
ax_inset.plot(x, droq, label="DroQ")
ax_inset.plot(x, sunrise, label="SUNRISE")
ax_inset.plot(x, random, label="Random")

ax_inset.set_xlim(4, 10)
ax_inset.set_ylim(12, 15)
ax_inset.set_xticks([4, 7, 10])
ax_inset.set_yticks([12, 15])
#ax_inset.grid()

ax_inset.set_title("Zoomed In", fontsize=10)

plt.savefig("Evaluation-Plots/deployed-agent-performance-with-inset-ces.pdf", transparent=True, bbox_inches="tight")
plt.show()