from matplotlib import pyplot as plt
import numpy as np

L = 1e6
T = 30

x = np.linspace(1, 30, 30)

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

sunrise = []

random = []

plt.hlines(np.log(L + 1), xmin = 0, xmax = T, color="black", linestyles="dashed")

plt.plot(x, redq, label="REDQ")
plt.plot(x, sbr, label="SBR")
plt.plot(x, droq, label="DroQ")
#plt.plot(x, sunrise, label="SUNRISE")
#plt.plot(x, random, label="Random")

#plt.fill_between(np.arange(1, T), 11.3622 - 0.01278, 11.3622 + 0.01278, alpha=0.5, label = "Standard Error")

plt.xlabel("Experiment")
plt.ylabel("sPCE")
plt.grid()
plt.legend()
plt.savefig("deployed-agent-performance.pdf")
plt.show()