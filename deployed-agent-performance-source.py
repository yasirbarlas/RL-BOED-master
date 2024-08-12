from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

L = 1e6
T = 30

x = np.linspace(1, T, T)

redq = [0.8222,  1.6816,  2.5744,  3.3894,  4.1136,  4.7610,  5.3569,  5.9341,
        6.5034,  7.0500,  7.5317,  7.9856,  8.4113,  8.7990,  9.1401,  9.4486,
        9.7469,  9.9922, 10.2120, 10.4269, 10.6073, 10.7753, 10.9348, 11.0745,
        11.1959, 11.3033, 11.4137, 11.5134, 11.6052, 11.6892]

sbr = [0.8039,  1.6529,  2.5173,  3.2934,  3.9913,  4.6359,  5.2140,  5.7756,
      6.3068,  6.8114,  7.2627,  7.6869,  8.0927,  8.4533,  8.7826,  9.0789,
      9.3720,  9.6215,  9.8405, 10.0535, 10.2373, 10.4095, 10.5709, 10.7171,
      10.8356, 10.9482, 11.0649, 11.1671, 11.2666, 11.3622]

droq = [0.8191,  1.6942,  2.5811,  3.3767,  4.1053,  4.7591,  5.3682,  5.9473,
        6.5118,  7.0490,  7.5221,  7.9756,  8.4033,  8.7774,  9.1177,  9.4253,
        9.7213,  9.9705, 10.1874, 10.3955, 10.5747, 10.7448, 10.9018, 11.0422,
        11.1661, 11.2820, 11.3985, 11.4988, 11.5947, 11.6800]

sunrise = [0.7856,  1.6404,  2.5233,  3.3307,  4.0579,  4.7467,  5.3697,  5.9775,
           6.5317,  7.0650,  7.5341,  7.9828,  8.4137,  8.8158,  9.1517,  9.4675,
           9.7627, 10.0127, 10.2636, 10.4910, 10.6815, 10.8694, 11.0285, 11.1780,
           11.3098, 11.4427, 11.5553, 11.6534, 11.7489, 11.8374]

random = [0.7989, 1.4600, 2.0370, 2.5387, 2.9800, 3.3669, 3.7184, 4.0286, 4.3166,
          4.5835, 4.8247, 5.0542, 5.2550, 5.4490, 5.6282, 5.7971, 5.9582, 6.1014,
          6.2471, 6.3774, 6.5057, 6.6296, 6.7486, 6.8600, 6.9671, 7.0731, 7.1730,
          7.2689, 7.3617, 7.4425]

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

inset_position = [0.5, 0.05, 2.6, 1.2]  # [x0, y0, width, height]
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

ax_inset.set_xlim(20, 30)
ax_inset.set_ylim(10, 12)
ax_inset.set_xticks([20, 25, 30])
ax_inset.set_yticks([10, 11, 12])
#ax_inset.grid()

ax_inset.set_title("Zoomed In", fontsize=10)

plt.savefig("Evaluation-Plots/deployed-agent-performance-with-inset-source.pdf", transparent=True, bbox_inches="tight")
plt.show()