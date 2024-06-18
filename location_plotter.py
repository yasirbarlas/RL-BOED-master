import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

theta1 = [-2.2659, -0.0179]
theta2 = [0.1618, -2.6031]

actions = [[ 0.0173, -0.0294], [-0.2860,  0.2704], [-0.4945, -0.0883], [-3.2703e-01, -1.3836e-01], [-1.8008e-01, -2.4261e-01],
           [ 3.4598e-01, -3.0162e-01], [ 0.2333,  0.4469], [-0.6549, -0.0104], [-0.5931,  0.0667], [-0.6085, -0.0308],
           [-1.3552e-01, -4.1013e-01], [-1.0959e-01, -5.1951e-01], [ 0.4911, -0.4772], [ 0.0081, -0.5687],
           [ 0.2182, -0.6119], [ 0.2470, -0.6688], [ 0.1187, -0.5168], [ 0.2982, -0.3432], [-0.3167, -0.6464],
           [-6.8605e-02, -5.0121e-01], [-0.4002,  0.0225], [-0.6882, -0.1161], [-0.2833, -0.5922], [-0.4571, -0.0102], 
           [ 0.5063, -0.4562], [-0.8994, -0.1439], [-0.5306,  0.1076], [-0.3953,  0.0456], [ 0.7144, -0.1330],
           [ 0.4066, -0.0981]]

#actions = [[-0.1127,  0.1377], [ 0.0338, -0.2059], [ 0.2735, -0.1603], [ 2.7681e-01,  1.0153e-01], 
#           [ 9.6857e-02, -1.7858e-01], [ 1.0092e-01, -1.6488e-01], [ 0.2376, -0.2565], [-0.0038, -0.1778],
#           [ 0.0007, -0.1719], [ 0.1262, -0.1619], [ 7.8014e-02, -1.7242e-01], [ 1.3675e-01, -1.6253e-01],
#           [ 0.0034, -0.1386], [ 0.1037, -0.0732], [ 0.0779, -0.1837], [ 0.5183, -0.2396], [ 0.4269, -0.0096],
#           [ 0.4075, -0.1671], [ 0.2789, -0.0221], [ 1.9606e-01, -1.3791e-01], [ 0.2895, -0.1443], [ 0.4035, -0.0956],
#           [ 0.5666, -0.0554], [ 0.4536, -0.1281], [ 0.5595, -0.0593], [ 0.3478, -0.0630], [ 0.4630, -0.4202],
#           [ 0.1019, -0.1961], [-0.0249, -0.2406], [ 0.7963, -0.1132]]

# Multiply each number by 4
scaled_actions = [[4 * x, 4 * y] for x, y in actions]

# Extracting x and y coordinates
x, y = zip(*scaled_actions)

# Points theta1 and theta2
theta1 = np.array([-2.2659, -0.0179])
theta2 = np.array([0.1618, -2.6031])

# Constants for the signal intensity equation
b = 1e-1
alpha1 = 1.0
alpha2 = 1.0
m = 1e-4

# Create a grid of points in the region around theta1 and theta2
grid_size = 100
xi = np.linspace(min(x) - 5, max(x) + 5, grid_size)
yi = np.linspace(min(y) - 5, max(y) + 5, grid_size)
xi, yi = np.meshgrid(xi, yi)

# Calculate the signal intensity at each point in the grid
intensity = b + (alpha1 / (m + (np.square(xi - theta1[0]) + np.square(yi - theta1[1])))) \
              + (alpha2 / (m + (np.square(xi - theta2[0]) + np.square(yi - theta2[1]))))

#cmap, norm = mcolors.from_levels_and_colors([0, 2, 5, 6], ['red', 'green', 'blue'])

# Plot the points and the intensity map
plt.figure(figsize=(10, 6))
scatter = plt.scatter(x, y, c=np.linspace(1, 30, len(scaled_actions)), cmap='viridis', label='Designs')
plt.colorbar(scatter, label='Experiment Order')
plt.contourf(xi, yi, intensity, levels=50, cmap='hot', alpha=0.4)

# Mark the points theta1 and theta2
plt.scatter([theta1[0], theta2[0]], [theta1[1], theta2[1]], color='blue', label='Locations', marker='x')

plt.xlim((-4, 4))
plt.ylim((-4, 4))

# Adding titles and labels
plt.title("Location Finding")
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
plt.legend()
plt.savefig("location_finding.pdf")
#plt.show()