import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# Seeds are 10, 50, 100 for evaluation time in select_policy_env.py
# Uncomment each set as required

# Example with K = 2 and SBR-430000
# thetas = [[-0.1029,  1.6810],
#           [-0.2708, -0.5432]]

# actions = [
#     [-0.1615, -0.0577],
#     [0.0015, 0.0909],
#     [-0.0035, -0.1472],
#     [0.0170, -0.1323],
#     [0.2143, -0.1227],
#     [-0.0869, -0.2274],
#     [-0.3041, 0.2217],
#     [-0.1891, -0.0173],
#     [0.0875, -0.0198],
#     [0.0169, -0.2105],
#     [0.0364, 0.4218],
#     [0.0431, 0.3167],
#     [0.0736, 0.3626],
#     [0.0659, 0.3029],
#     [0.0405, 0.2772],
#     [-0.0140, -0.0089],
#     [-0.0126, 0.2295],
#     [0.0638, 0.3026],
#     [0.0844, 0.2167],
#     [0.0781, 0.3190],
#     [-0.0246, -0.1002],
#     [0.0122, 0.3424],
#     [0.1959, 0.2081],
#     [0.1039, 0.3588],
#     [-0.0244, 0.3871],
#     [-0.0591, -0.0745],
#     [-0.1346, -0.1314],
#     [0.0942, 0.1207],
#     [-0.1023, 0.1368],
#     [0.1553, 0.0805]
# ]

# Reward in sPCE
#torch.tensor([[ 0.6238,  1.3447,  2.8683,  3.4132,  3.9744,  5.0533,  5.7035,  5.9245,
#          5.8394,  6.0645,  9.3719,  9.8576,  9.7412,  9.7257,  9.7295,  8.4801,
#          8.6795,  8.7394,  8.9167,  8.0671,  8.6255,  8.3292,  9.0653,  7.9476,
#         11.8634, 12.5559, 12.2715, 12.2716, 12.2481, 12.2560]],
#       device='cuda:0')

####################################################

# Example with K = 2 and SBR-430000
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784]]

# actions = [
#     [-0.0831, -0.1125],
#     [0.0977, -0.0256],
#     [-0.0951, 0.0449],
#     [0.1610, -0.2650],
#     [0.2937, 0.1274],
#     [-0.1106, -0.2927],
#     [-0.1389, -0.2152],
#     [-0.2225, -0.1577],
#     [-0.1525, 0.3058],
#     [0.0390, -0.3362],
#     [-0.2529, -0.1104],
#     [-0.1922, 0.1335],
#     [0.3991, -0.5619],
#     [-0.0411, -0.3969],
#     [-0.0445, -0.3719],
#     [0.3765, -0.3403],
#     [0.4118, -0.3577],
#     [0.4084, -0.3037],
#     [0.4116, -0.2571],
#     [0.3062, -0.3103],
#     [0.3661, -0.3427],
#     [0.1196, -0.3722],
#     [0.3786, -0.2262],
#     [0.1672, -0.4263],
#     [0.4698, -0.4587],
#     [0.2574, -0.0759],
#     [-0.6178, 0.3035],
#     [0.3328, 0.0172],
#     [-0.0744, -0.2074],
#     [-0.1257, -0.4725]
# ]

# Reward in sPCE
#torch.tensor([[ 0.3749,  0.0379,  0.8097,  1.6274,  2.4433,  2.6763,  3.4873,  3.5341,
#          4.2923,  5.6734,  5.8722,  6.5584,  7.7322,  7.5949,  7.7174, 10.2024,
#         10.8568, 11.5068, 11.4828, 11.4210, 12.3657, 12.7867, 12.8183, 12.7592,
#         12.7735, 12.7651, 12.7745, 12.8593, 13.1479, 13.1506]],
#       device='cuda:0')

####################################################

# Example with K = 2 and SBR-430000
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402]]

# actions = [
#     [0.0332, -0.1990],
#     [0.0055, 0.1055],
#     [-0.0038, 0.0238],
#     [-0.0828, 0.0509],
#     [-0.0295, 0.0277],
#     [-0.0699, -0.0106],
#     [-0.0548, -0.0045],
#     [-0.0079, -0.0943],
#     [-0.0657, 0.0276],
#     [0.2659, 0.2049],
#     [-0.1810, 0.0999],
#     [0.0370, 0.2726],
#     [0.2423, -0.1290],
#     [-0.4388, -0.0014],
#     [-0.4595, -0.1745],
#     [-0.2852, 0.3570],
#     [-0.2082, 0.3503],
#     [-0.2462, 0.2715],
#     [-0.1409, 0.1856],
#     [-0.3080, 0.2352],
#     [-0.2474, 0.2341],
#     [-0.2077, 0.2476],
#     [-0.1981, 0.2997],
#     [-0.1751, 0.3004],
#     [-0.2140, 0.1250],
#     [-0.2440, 0.3792],
#     [-0.3555, 0.2623],
#     [-0.1717, 0.2927],
#     [-0.2491, 0.2708],
#     [-0.3195, 0.2191]
# ]

# Reward in sPCE
#torch.tensor([[-0.0500,  1.8614,  2.6655,  3.5680,  1.9288,  2.4695,  2.3376,  2.3848,
#          2.2426,  2.3558,  2.8619,  3.1630,  3.7957,  4.2676,  4.4318,  7.1295,
#          7.4915,  7.9225,  8.4439,  8.5971,  8.3123,  8.0100,  7.9270,  8.2219,
#          8.2496,  8.4787,  9.4624,  9.4159,  9.5310,  9.6201]],
#       device='cuda:0')

####################################################

# Example with K = 1 and SBR-430000
# thetas = [[ 1.4350, -1.4982]]

# actions = [
#     [-0.0901, -0.0345],
#     [0.4153, 0.0364],
#     [0.2941, -0.2593],
#     [0.1818, -0.2774],
#     [0.2340, 0.4709],
#     [-0.3737, 0.2442],
#     [-0.0172, -0.5739],
#     [-0.4782, -0.4299],
#     [0.4060, -0.4981],
#     [0.3287, -0.4374],
#     [0.4288, -0.4201],
#     [0.3485, -0.4217],
#     [0.3458, -0.5435],
#     [0.2873, 0.1424],
#     [0.2876, -0.6108],
#     [-0.2051, -0.1115],
#     [0.0908, -0.3449],
#     [0.6649, -0.3230],
#     [0.3764, 0.3412],
#     [0.0428, -0.2601],
#     [0.3172, -0.4081],
#     [-0.1713, 0.0032],
#     [0.2721, 0.3059],
#     [-0.1249, 0.1204],
#     [0.3570, -0.3925],
#     [-0.3825, 0.6573],
#     [0.4696, 0.2921],
#     [0.6570, -0.3807],
#     [-0.7384, -0.5029],
#     [0.6016, -0.6807]
# ]

# Reward in sPCE
#torch.tensor([[1.2810, 2.2714, 4.3328, 4.7918, 4.8134, 4.8097, 5.1408, 5.1696, 5.6653,
#         5.9812, 7.0184, 6.6874, 6.4453, 6.4645, 6.8724, 6.8617, 6.9051, 6.8897,
#         6.9477, 6.9412, 7.8424, 7.8459, 7.7957, 7.8060, 8.7553, 8.7642, 8.7735,
#         8.7016, 8.7052, 8.8889]], device='cuda:0')

####################################################

# # Example with K = 1 and SBR-430000
# thetas = [[-1.4033,  1.8134]]

# actions = [
#     [0.1079, 0.0939],
#     [-0.2928, -0.0958],
#     [-0.0901, -0.4150],
#     [-0.3335, 0.3247],
#     [-0.2502, 0.2751],
#     [0.4592, -0.1783],
#     [-0.2783, 0.2084],
#     [-0.1212, 0.4919],
#     [-0.1235, 0.4130],
#     [0.0386, 0.5406],
#     [-0.6301, 0.0964],
#     [-0.4918, -0.5530],
#     [-0.4532, 0.5997],
#     [-0.4466, 0.3125],
#     [-0.3647, 0.4197],
#     [0.1852, -0.4169],
#     [0.0502, 0.2600],
#     [0.2000, 0.2752],
#     [-0.3599, 0.4094],
#     [-0.6058, -0.3251],
#     [0.5107, -0.4540],
#     [-0.2163, 0.7720],
#     [-0.5447, 0.7043],
#     [-0.1555, 0.4275],
#     [-0.1526, 0.4915],
#     [-0.1161, 0.5218],
#     [-0.3961, 0.5365],
#     [-0.6998, 0.4656],
#     [-0.4833, 0.0097],
#     [-0.3352, 0.0382]
# ]

# Reward in sPCE
#torch.tensor([[1.1946, 1.5449, 2.2570, 3.6841, 4.3630, 4.3716, 5.2211, 5.8736, 6.3663,
#         6.5318, 6.5712, 6.5729, 6.7735, 6.3624, 8.1950, 8.1748, 8.1375, 8.1703,
#         8.6715, 8.6654, 8.6694, 8.7210, 8.7079, 8.6834, 8.7092, 8.7719, 8.7689,
#         8.7857, 8.7913, 8.8918]], device='cuda:0')

####################################################

# Example with K = 1 and SBR-430000
# thetas = [[-0.1029,  1.6810]]

# actions = [
#     [0.0474, -0.1059],
#     [-0.0237, 0.2996],
#     [0.0289, 0.2623],
#     [-0.3088, 0.2788],
#     [0.2100, 0.3544],
#     [0.6007, 0.0062],
#     [-0.3618, -0.2823],
#     [-0.6813, 0.2448],
#     [-0.1853, 0.5929],
#     [-0.4611, 0.4271],
#     [-0.1996, -0.7933],
#     [0.6816, -0.0405],
#     [-0.2277, 0.3885],
#     [-0.3004, 0.5073],
#     [0.3290, 0.5553],
#     [0.2925, -0.0164],
#     [-0.4141, 0.3459],
#     [0.3951, 0.6454],
#     [-0.7165, 0.2673],
#     [0.6795, 0.0682],
#     [-0.2072, -0.7965],
#     [-0.2650, 0.4442],
#     [-0.6365, 0.2059],
#     [0.3362, 0.3321],
#     [0.0474, 0.7695],
#     [-0.4789, 0.5226],
#     [-0.5893, 0.0872],
#     [0.7731, -0.2059],
#     [0.4392, -0.7406],
#     [-0.4445, -0.8601]
# ]

# Reward in sPCE
#torch.tensor([[1.0210, 3.3252, 3.1300, 3.9492, 4.0898, 3.9377, 3.9182, 3.6850, 4.6885,
#         4.7674, 4.7641, 4.7587, 5.3241, 5.4333, 5.4917, 5.5509, 5.6078, 5.6625,
#         5.6518, 5.6117, 5.6140, 5.6681, 5.6574, 5.7501, 5.7948, 5.5396, 5.4763,
#         5.4362, 5.4371, 5.4481]], device='cuda:0')

####################################################

# Example with K = 3 and SBR-430000
# thetas = [[-0.1029,  1.6810],
#           [-0.2708, -0.5432],
#           [-0.3521,  0.3413]]

# actions = [
#     [-0.0836, 0.1640],
#     [-0.0432, 0.1389],
#     [0.0647, -0.1104],
#     [-0.1938, -0.2200],
#     [-0.1080, 0.0319],
#     [-0.0807, -0.0234],
#     [-0.0964, 0.0387],
#     [-0.0225, 0.0800],
#     [-0.1401, 0.0859],
#     [-0.0111, 0.0155],
#     [-0.0206, 0.0256],
#     [-0.2056, -0.0086],
#     [-0.1015, 0.0869],
#     [-0.1151, -0.3266],
#     [-0.1565, -0.1014],
#     [-0.2301, -0.1519],
#     [-0.0163, -0.1472],
#     [-0.0215, 0.0519],
#     [-0.1306, 0.0563],
#     [-0.1420, -0.0449],
#     [0.0561, -0.0474],
#     [-0.0005, -0.1439],
#     [-0.1559, -0.2825],
#     [-0.0061, -0.1968],
#     [-0.0680, -0.1997],
#     [-0.0254, 0.1408],
#     [-0.0645, -0.1886],
#     [0.0049, -0.1254],
#     [-0.2864, 0.1590],
#     [-0.2102, 0.0426]
# ]

# Reward in sPCE
#torch.tensor([[ 0.2431,  0.4206, -2.3082, -1.5100,  0.2622,  0.5455,  0.8500,  2.0984,
#          2.4589,  2.8410,  2.7356,  3.2386,  3.5757,  4.2360,  5.3037,  5.4802,
#          9.4258,  9.1452,  7.8147,  7.6880,  7.7318,  8.1042,  8.3075,  8.6513,
#          8.7703,  9.0925, 10.1000, 10.3050, 10.7361, 10.7017]],
#       device='cuda:0')

####################################################

# Example with K = 3 and SBR-430000
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402],
#           [-2.1387, -0.6234]]

# actions = [
#     [-0.1302, -0.0600],
#     [-0.0786, -0.0031],
#     [-0.0135, 0.0839],
#     [-0.0213, 0.0497],
#     [0.0273, 0.0304],
#     [-0.1016, -0.1073],
#     [0.1131, 0.0455],
#     [-0.2642, -0.2799],
#     [-0.1088, -0.3703],
#     [0.0012, -0.1541],
#     [-0.2265, 0.0165],
#     [0.0390, -0.2760],
#     [-0.5492, -0.2912],
#     [-0.3848, -0.1951],
#     [-0.3527, -0.0634],
#     [-0.4088, -0.0818],
#     [-0.2816, -0.1350],
#     [-0.2532, -0.0380],
#     [-0.2499, -0.0426],
#     [-0.2829, -0.0453],
#     [-0.3782, -0.0633],
#     [-0.3247, -0.0990],
#     [-0.3132, -0.2180],
#     [-0.3604, 0.0686],
#     [-0.3275, 0.0801],
#     [-0.5233, 0.0317],
#     [-0.4244, -0.0692],
#     [-0.3767, 0.1626],
#     [-0.4556, 0.0524],
#     [-0.3583, 0.0875]
# ]

# Reward in sPCE
#torch.tensor([[ 0.5472,  1.8477,  2.8792,  3.9853,  5.1516,  5.4352,  6.3135,  6.5275,
#          6.2431,  6.0818,  7.0097,  7.9464,  9.1966,  9.9568, 10.6195, 10.9833,
#         10.9366, 10.9752, 11.1468, 11.2311, 11.4685, 11.5094, 11.7203, 11.6122,
#         11.8353, 12.0566, 11.8398, 11.9788, 11.7159, 11.7677]],
#       device='cuda:0')

####################################################

# Example with K = 3 and SBR-430000
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784],
#           [-2.6479, -0.0233]]

# actions = [
#     [-0.0535, 0.1670],
#     [-0.0094, -0.2460],
#     [0.0632, -0.0844],
#     [0.0273, -0.1414],
#     [-0.1701, -0.1298],
#     [0.2601, -0.0397],
#     [-0.2039, -0.0641],
#     [0.4036, 0.0073],
#     [-0.3690, -0.1522],
#     [-0.2273, -0.2669],
#     [0.1447, 0.0623],
#     [0.0111, -0.3589],
#     [-0.1407, -0.3410],
#     [-0.1009, -0.3841],
#     [-0.1686, -0.3219],
#     [-0.1599, -0.2854],
#     [-0.3347, -0.1483],
#     [-0.1641, 0.0485],
#     [-0.1854, 0.1689],
#     [0.0425, 0.0483],
#     [-0.0046, -0.3341],
#     [-0.2726, -0.3220],
#     [-0.1565, -0.3382],
#     [-0.1686, -0.1661],
#     [-0.2486, -0.2807],
#     [-0.1079, -0.1988],
#     [-0.2219, -0.4014],
#     [-0.0032, -0.6491],
#     [-0.4642, -0.1256],
#     [-0.1829, -0.4619]
# ]

# Reward in sPCE
#torch.tensor([[2.2746, 3.0880, 2.7747, 2.8143, 3.4129, 3.8309, 4.6540, 5.3093, 3.7132,
#         5.0742, 5.2315, 4.9594, 5.7421, 6.4166, 7.2278, 7.5851, 8.3825, 8.5720,
#         8.4814, 8.4930, 8.1948, 8.7279, 8.4105, 8.4586, 8.8356, 9.1070, 8.7948,
#         9.1602, 9.6978, 9.6347]], device='cuda:0')

####################################################
[]
# Example with K = 4 and SBR-430000
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784],
#           [-2.6479, -0.0233],
#           [-1.3761,  1.1551]]

# actions = [
#     [0.0876, -0.1446],
#     [-0.0274, 0.1283],
#     [-0.1898, -0.0238],
#     [0.2931, 0.0943],
#     [-0.3363, -0.0004],
#     [-0.0667, -0.3604],
#     [-0.1289, 0.3407],
#     [-0.1826, 0.2263],
#     [-0.0385, -0.2863],
#     [-0.0545, 0.2810],
#     [-0.1304, -0.1114],
#     [-0.2199, 0.3264],
#     [-0.1849, 0.0522],
#     [-0.0960, -0.1287],
#     [-0.1286, 0.2038],
#     [0.0424, 0.3586],
#     [-0.2764, 0.1936],
#     [-0.2694, 0.2124],
#     [-0.3007, 0.2643],
#     [-0.2706, -0.1415],
#     [-0.3177, 0.3220],
#     [-0.0215, -0.4496],
#     [-0.1399, -0.3725],
#     [-0.2026, -0.0054],
#     [-0.3134, -0.0018],
#     [-0.4298, -0.1745],
#     [-0.2799, -0.0074],
#     [-0.3467, -0.1525],
#     [-0.1880, 0.3967],
#     [-0.4528, -0.2626]
# ]

# Reward in sPCE
#torch.tensor([[ 1.7766,  3.0959,  3.6874,  4.6594,  4.7923,  5.4300,  5.4461,  6.0523,
#          6.0915,  6.9440,  6.4401,  6.9091,  6.5140,  6.9979,  6.8499,  7.4413,
#          8.8539,  8.5364, 10.0561, 10.7817, 11.5169, 11.9974, 11.7412, 11.8136,
#         11.6901, 11.9979, 11.2443, 11.1109, 11.0036, 11.1013]],
#       device='cuda:0')

####################################################

# Example with K = 4 and SBR-430000
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402],
#           [-2.1387, -0.6234],
#           [-0.4541, -0.6023]]

# actions = [
#     [0.0131, -0.4231],
#     [-0.1000, -0.1984],
#     [-0.0321, -0.1095],
#     [-0.1257, -0.1197],
#     [0.0491, -0.0712],
#     [0.3010, 0.1451],
#     [0.0593, 0.1609],
#     [-0.1146, 0.1319],
#     [-0.1079, 0.0807],
#     [-0.0825, 0.0520],
#     [-0.1097, -0.0535],
#     [-0.0295, -0.1364],
#     [-0.1070, -0.0938],
#     [-0.0411, -0.0859],
#     [-0.0620, -0.0276],
#     [-0.0998, 0.0326],
#     [-0.0886, 0.0408],
#     [-0.0415, -0.1524],
#     [-0.0592, 0.1129],
#     [-0.2053, -0.0414],
#     [0.0725, 0.0029],
#     [-0.0249, -0.0011],
#     [-0.1063, -0.0705],
#     [-0.1200, -0.1593],
#     [-0.1377, -0.1043],
#     [-0.2242, -0.0272],
#     [0.1232, 0.0658],
#     [-0.1572, 0.2453],
#     [-0.1223, 0.0367],
#     [-0.0668, -0.1221]
# ]

# Reward in sPCE
#tensor([[ 0.3554,  1.4530,  2.1375,  3.7168,  4.2600,  5.0677,  5.6481,  6.5644,
#          7.2907,  9.6251,  9.5506,  9.6579,  9.5943,  9.3948,  8.8852,  9.3487,
#         10.3067, 10.7227, 10.9487, 11.3395, 11.2130, 11.8867, 12.3056, 13.2429,
#         13.3380, 13.3577, 13.4994, 13.2675, 13.1984, 13.1971]],
#       device='cuda:0')

####################################################

# Example with K = 4 and SBR-430000
# thetas = [[-0.1029,  1.6810],
#           [-0.2708, -0.5432],
#           [-0.3521,  0.3413],
#           [-1.6456,  1.5698]]

# actions = [
#     [-0.0635, 0.1350],
#     [0.0023, 0.1069],
#     [-0.0101, 0.1007],
#     [-0.1014, 0.1210],
#     [0.1566, -0.2621],
#     [0.0191, 0.0017],
#     [-0.0199, 0.0456],
#     [-0.0449, 0.1246],
#     [0.1410, -0.0555],
#     [0.0504, 0.1120],
#     [0.0341, 0.1299],
#     [0.0304, 0.0829],
#     [-0.0240, 0.1147],
#     [0.1472, 0.1271],
#     [-0.1448, 0.1880],
#     [0.4049, 0.0035],
#     [-0.3048, 0.0306],
#     [-0.2900, 0.0032],
#     [-0.1663, -0.0777],
#     [-0.2220, -0.0613],
#     [-0.1290, -0.0790],
#     [-0.1761, -0.0458],
#     [-0.0245, -0.0374],
#     [0.0288, 0.0369],
#     [-0.0496, 0.0898],
#     [-0.0398, 0.0966],
#     [-0.0294, -0.0671],
#     [-0.1797, 0.0859],
#     [-0.1382, -0.0715],
#     [0.0099, 0.1493]
# ]

# Reward in sPCE
#torch.tensor([[2.0584, 2.3517, 2.7282, 3.8823, 4.2012, 4.3956, 4.1294, 4.2512, 3.2189,
#         3.7609, 3.7995, 4.5785, 4.5816, 4.6611, 5.6128, 6.6045, 5.6372, 5.1840,
#         5.8358, 5.5399, 5.9500, 5.8346, 6.5847, 6.0391, 5.6129, 5.3912, 5.6465,
#         5.8603, 5.9955, 6.0693]], device='cuda:0')

####################################################

# Example with K = 5 and SBR-430000
# thetas = [[-0.1029,  1.6810],
#           [-0.2708, -0.5432],
#           [-0.3521,  0.3413],
#           [-1.6456,  1.5698],
#           [-0.2490, -0.4059]]

# actions = [
#     [-0.0527, 0.1313],
#     [-0.0246, 0.0953],
#     [0.0416, 0.0424],
#     [0.0188, 0.0776],
#     [-0.2408, -0.0836],
#     [0.0364, -0.0062],
#     [-0.1262, 0.0817],
#     [-0.0422, 0.0767],
#     [-0.0155, 0.0545],
#     [-0.0034, 0.0984],
#     [0.0157, 0.0830],
#     [-0.0965, 0.1381],
#     [0.1297, 0.0197],
#     [0.0197, 0.1632],
#     [-0.1585, 0.1568],
#     [-0.0484, 0.0777],
#     [-0.0942, 0.0885],
#     [0.2044, 0.1080],
#     [-0.0599, 0.0735],
#     [0.2616, 0.0461],
#     [0.3217, 0.2032],
#     [0.1840, 0.2719],
#     [0.1633, 0.2430],
#     [0.1520, -0.0615],
#     [-0.1241, 0.1724],
#     [-0.0482, -0.0924],
#     [0.2095, 0.3022],
#     [-0.1693, 0.2873],
#     [-0.0235, 0.2377],
#     [0.0518, 0.2885]
# ]

# Reward in sPCE
#torch.tensor([[ 1.6381,  1.7073,  1.7525,  1.1354,  1.3995,  0.3564,  1.8859,  2.5307,
#          3.1158,  2.2284,  2.1973,  2.8212,  5.2330,  4.7959,  4.5715,  4.5577,
#          6.5460,  6.8709,  7.2833,  7.3852,  7.1687,  7.4428,  7.4362,  7.6272,
#          7.5821, 10.6608, 10.7014, 10.8276, 10.8716, 10.9488]],
#       device='cuda:0')

####################################################

# Example with K = 5 and SBR-430000
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784],
#           [-2.6479, -0.0233],
#           [-1.3761,  1.1551],
#           [-1.0914, -0.3823]]

# actions = [
#     [0.0408, 0.0018],
#     [-0.2043, -0.2073],
#     [-0.0432, -0.1836],
#     [-0.1773, -0.0968],
#     [-0.1813, -0.0634],
#     [-0.1301, -0.0103],
#     [0.3382, -0.1861],
#     [0.2944, 0.0584],
#     [0.0381, -0.0170],
#     [-0.0369, -0.0157],
#     [0.1960, -0.0054],
#     [0.2210, -0.3321],
#     [0.1106, -0.2887],
#     [0.0876, -0.3308],
#     [0.0775, -0.3310],
#     [0.1338, -0.2469],
#     [-0.1528, -0.1599],
#     [0.0696, -0.2612],
#     [-0.0295, -0.2622],
#     [0.1278, -0.3511],
#     [0.0470, -0.2301],
#     [-0.0847, -0.2248],
#     [-0.1618, -0.3814],
#     [-0.0341, -0.2171],
#     [0.3076, -0.3135],
#     [0.5140, -0.2658],
#     [0.4249, -0.4278],
#     [0.4666, -0.3475],
#     [0.6144, -0.3129],
#     [0.4085, -0.4268]
# ]

# Reward in sPCE
#torch.tensor([[ 1.8995,  1.8943,  2.3846,  3.1608,  3.9011,  4.6303,  5.3941,  7.1861,
#          7.4097,  7.5934,  7.6676,  8.0933,  8.4453,  8.6992,  8.7828,  9.0448,
#          9.2828,  9.1850,  9.4368,  9.2069,  9.3750,  8.8716, 10.2218,  9.1556,
#         10.1731,  9.5720, 10.8832, 11.6019, 11.9858, 12.2705]],
#       device='cuda:0')

####################################################

# Example with K = 5 and SBR-430000
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402],
#           [-2.1387, -0.6234],
#           [-0.4541, -0.6023],
#           [ 0.3330, -2.1362]]

# actions = [
#     [-0.1801, -0.1450],
#     [-0.0683, -0.0588],
#     [0.0070, -0.0471],
#     [-0.0736, -0.0154],
#     [-0.1067, 0.0506],
#     [-0.1187, 0.0440],
#     [-0.0305, 0.0407],
#     [-0.1561, -0.0839],
#     [-0.0237, -0.1906],
#     [-0.1332, -0.1333],
#     [0.0002, -0.1244],
#     [-0.0408, -0.0648],
#     [-0.1061, 0.0188],
#     [-0.1049, 0.0445],
#     [-0.0622, -0.1986],
#     [-0.0953, 0.1147],
#     [-0.2780, -0.0722],
#     [0.0585, -0.0216],
#     [-0.0637, -0.0210],
#     [-0.1550, -0.0993],
#     [-0.1694, -0.1988],
#     [-0.1508, -0.1274],
#     [-0.2056, -0.0633],
#     [0.0584, -0.0007],
#     [-0.1535, 0.1477],
#     [-0.1222, -0.0080],
#     [-0.0784, -0.1214],
#     [-0.1390, -0.0251],
#     [-0.1451, -0.1788],
#     [-0.0586, -0.1560]
# ]

# Reward in sPCE
#torch.tensor([[ 1.5792,  1.9126,  2.2945,  3.0473,  4.4749,  5.0626,  6.8582,  7.5306,
#          7.7637,  9.7230,  9.5173,  9.1746,  9.2353,  9.2839,  9.5100,  9.7899,
#          9.9377,  9.8699, 10.2765, 10.8725, 11.0350, 11.3902, 11.4469, 12.3562,
#         12.6753, 12.6652, 12.7638, 12.8588, 12.7942, 12.8102]],
#       device='cuda:0')

####################################################

print(len(actions))

# Multiply each number by 4
scaled_actions = [[4 * x, 4 * y] for x, y in actions]

# Extracting x and y coordinates
x, y = zip(*scaled_actions)

# Points theta1 and theta2
#theta1 = np.array([-2.2659e+00, -1.7917e-02])
#theta2 = np.array([ 1.6182e-01, -2.6031e+00])

#theta1 = np.array([ 1.8805e+00, -1.1253e+00])
#theta2 = np.array([ 1.8882e-01, -7.5086e-01])

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
#intensity = b + (alpha1 / (m + (np.square(xi - theta_single_k1[0] + np.square(yi - theta_single_k1[1]))) 
              #+ (alpha2 / (m + (np.square(xi - theta2[0] + np.square(yi - theta2[1])))

#cmap, norm = mcolors.from_levels_and_colors[0, 2, 5, 6], ['red', 'green', 'blue']

# Plot setup
fig, ax = plt.subplots(figsize=(8, 4.8))
scatter = ax.scatter(x, y, c=np.linspace(1, 30, len(scaled_actions)), cmap='viridis', label='Designs')
ax.scatter([i for i, j in thetas], [j for i, j in thetas], color='blue', label='Objects', marker='x')
ax.set_xlim((-4, 4))
ax.set_ylim((-4, 4))
ax.set_title(f"2D Location Finding with {len(thetas)} Objects")
ax.legend()

# Add colorbar below the plot
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', fraction=0.03, pad=0.08)
cbar.set_label('Experiment Order')

plt.xlim((-4, 4))
plt.ylim((-4, 4))

# Adding titles and labels
plt.title(f"2D Location Finding with {len(thetas)} Objects")
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
plt.legend()
plt.savefig(f"Location-Finding-Plots/location_finding_sbr_430000_k_{len(thetas)}.pdf", transparent=True, bbox_inches="tight")
#plt.show()