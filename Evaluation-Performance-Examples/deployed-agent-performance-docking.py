from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

L = 1e6
T = 20

x = np.linspace(1, T, T)

redq = [0.0223, 0.0401, 0.0531, 0.0707, 0.0899, 0.1059, 0.1198, 0.1413, 0.1642,
        0.1781, 0.1976, 0.2035, 0.2194, 0.2316, 0.2465, 0.2605, 0.2791, 0.2962,
        0.3077, 0.3217]

sbr = [0.0263, 0.0414, 0.0579, 0.0691, 0.0850, 0.0920, 0.1061, 0.1198, 0.1306,
       0.1384, 0.1472, 0.1553, 0.1641, 0.1792, 0.1913, 0.2015, 0.2125, 0.2224,
       0.2310, 0.2426]

droq = [0.0226, 0.0390, 0.0518, 0.0680, 0.0884, 0.1046, 0.1187, 0.1398, 0.1620,
        0.1762, 0.1961, 0.2022, 0.2181, 0.2310, 0.2468, 0.2619, 0.2799, 0.2971,
        0.3093, 0.3231]

sunrise = [0.0153, 0.0351, 0.0465, 0.0610, 0.0807, 0.1018, 0.1184, 0.1308, 0.1462,
           0.1648, 0.1786, 0.1919, 0.2059, 0.2167, 0.2288, 0.2425, 0.2563, 0.2752,
           0.2916, 0.3057]

fig, ax = plt.subplots(figsize=(8, 4.8))
#fig, ax = plt.subplots()
#ax.hlines(np.log(L + 1), xmin = 0, xmax = T, color="black", linestyles="dashed")
ax.plot(x, redq, label="REDQ")
ax.plot(x, sbr, label="SBR")
ax.plot(x, droq, label="DroQ")
ax.plot(x, sunrise, label="SUNRISE")
ax.set_xlim(0, T)
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.set_xlabel("Experiment")
ax.set_ylabel("sPCE")
ax.set_title("Average sPCE for Deployed Agents")
ax.grid()
ax.legend()

inset_position = [0.5, 0.05, 2.6, 1.1]  # [x0, y0, width, height]
ax_inset = inset_axes(ax, width=inset_position[2], height=inset_position[3], loc='lower left',
                      bbox_to_anchor=(inset_position[0], inset_position[1], 
                                      inset_position[2], inset_position[3]), 
                      bbox_transform=ax.transAxes)
#ax_inset.hlines(np.log(L + 1), xmin = 0, xmax = T, color="black", linestyles="dashed")
ax_inset.plot(x, redq, label="REDQ")
ax_inset.plot(x, sbr, label="SBR")
ax_inset.plot(x, droq, label="DroQ")
ax_inset.plot(x, sunrise, label="SUNRISE")

ax_inset.set_xlim(15, 20)
ax_inset.set_ylim(0.2, 0.4)
ax_inset.set_xticks([14, 17, 20])
ax_inset.set_yticks([0.2, 0.3, 0.4])
#ax_inset.grid()

ax_inset.set_title("Zoomed In", fontsize=10)

plt.savefig("Evaluation-Plots/deployed-agent-performance-with-inset-docking.pdf", transparent=True, bbox_inches="tight")
plt.show()