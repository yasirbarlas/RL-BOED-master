import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# Seeds are 10, 50, 100 for evaluation time in select_policy_env.py
# Uncomment each set as required

# Example with K = 2 and REDQ-Blau
# thetas = [[-0.1029,  1.6810], [-0.2708, -0.5432]]

# actions = [
#     [-0.1287, -0.0653],
#     [-0.0437,  0.0738],
#     [ 0.0364, -0.1112],
#     [ 0.0986,  0.0007],
#     [-0.0898, -0.1862],
#     [-0.0966, -0.1710],
#     [-0.1240, -0.1534],
#     [-0.1067, -0.1181],
#     [-0.0921, -0.1731],
#     [ 0.2009,  0.3579],
#     [ 0.3360,  0.1063],
#     [ 0.3387,  0.1979],
#     [-0.1237,  0.3749],
#     [-0.1351,  0.2728],
#     [-0.1630,  0.2599],
#     [-0.1476,  0.2799],
#     [-0.2368,  0.2976],
#     [-0.1852,  0.3332],
#     [-0.1671,  0.2986],
#     [-0.1568,  0.2900],
#     [-0.1707,  0.2382],
#     [-0.2216,  0.3213],
#     [-0.0070,  0.4200],
#     [-0.0351, -0.1244],
#     [-0.2482,  0.1576],
#     [-0.0975, -0.1410],
#     [-0.2282,  0.1323],
#     [-0.0952,  0.0391],
#     [-0.2070,  0.0280],
#     [-0.0766, -0.0089]
# ]

# Reward in sPCE
#torch.tensor([[ 0.9575,  1.6191,  2.9012,  3.5540,  4.6299,  5.1495,  5.8848,  6.9085,
#          6.8845,  7.9263,  8.1692,  8.0527,  8.9844,  8.6768,  9.9663,  8.7120,
#          8.8956,  9.0205, 10.0993,  8.8843,  8.7390,  9.3027, 11.4235, 12.1972,
#         12.1572, 12.0522, 12.0318, 12.0546, 12.1036, 12.1559]],
#       device='cuda:0')

####################################################

# Example with K = 2 and REDQ-Blau
# thetas = [[ 1.4350, -1.4982], [-0.8099, -1.1784]]

# actions = [
#     [-0.0731, -0.1037],
#     [ 0.0419,  0.0700],
#     [ 0.1450, -0.2104],
#     [ 0.2649, -0.1015],
#     [-0.1649, -0.1397],
#     [-0.0180, -0.3395],
#     [-0.1814,  0.0702],
#     [ 0.1922,  0.3057],
#     [-0.3888, -0.0298],
#     [-0.2003, -0.4138],
#     [-0.2406, -0.3605],
#     [-0.3070, -0.3314],
#     [-0.0533, -0.3960],
#     [-0.4644, -0.4064],
#     [ 0.0804, -0.4987],
#     [-0.1597, -0.3576],
#     [ 0.0772, -0.4170],
#     [-0.3393, -0.3556],
#     [ 0.5426, -0.1795],
#     [ 0.2181, -0.0974],
#     [-0.0068,  0.0260],
#     [-0.0262, -0.1911],
#     [ 0.3955,  0.2681],
#     [-0.2732, -0.2952],
#     [ 0.4992, -0.2946],
#     [ 0.5304, -0.3019],
#     [ 0.3947, -0.2584],
#     [ 0.4583, -0.3422],
#     [ 0.4361, -0.3850],
#     [ 0.4380, -0.4436]
# ]

# Reward in sPCE
#torch.tensor([[ 0.3167,  0.0161,  1.0029,  1.4518,  1.9943,  2.8899,  3.8478,  5.1987,
#          6.1144,  7.0561,  7.8333,  7.9693,  8.1442,  8.2216,  8.6607,  8.7836,
#          8.9684,  8.7720,  9.2571,  9.1216,  9.1523,  9.3875,  9.4508,  9.8294,
#          9.3062, 10.0352, 11.1182, 11.2793, 11.7717, 11.8174]],
#       device='cuda:0')

####################################################

# Example with K = 2 and REDQ-Blau
# thetas = [[-1.4033,  1.8134], [-0.2177,  0.2402]]

# actions = [
#     [ 0.0090, -0.1647],
#     [-0.0038,  0.1057],
#     [ 0.0059,  0.0326],
#     [-0.0212,  0.0628],
#     [-0.0333,  0.0335],
#     [-0.1154, -0.0072],
#     [ 0.1323, -0.0112],
#     [-0.2471, -0.2556],
#     [-0.1689,  0.0724],
#     [ 0.0048,  0.3053],
#     [ 0.1995,  0.0193],
#     [-0.3165,  0.0031],
#     [ 0.2869, -0.4008],
#     [-0.1220,  0.4332],
#     [ 0.1069,  0.4284],
#     [-0.3511,  0.3563],
#     [-0.2907,  0.2904],
#     [-0.3027,  0.2392],
#     [-0.2363,  0.1909],
#     [-0.2894,  0.2620],
#     [-0.2453,  0.2381],
#     [-0.2584,  0.2588],
#     [-0.2744,  0.2816],
#     [-0.2657,  0.2894],
#     [-0.2662,  0.1592],
#     [-0.3012,  0.4324],
#     [-0.3512,  0.3396],
#     [-0.1833,  0.3853],
#     [-0.2508,  0.3795],
#     [-0.3022,  0.3151]
# ]

# Reward in sPCE
#torch.tensor([[ 0.0499,  2.1287,  2.8529,  3.3964,  1.8369,  2.2504,  2.7576,  3.0045,
#          3.6795,  3.8395,  4.0040,  4.1007,  5.0002,  5.7487,  6.0862,  7.7763,
#          8.5525,  8.5309,  8.9328,  8.6345,  8.2891,  7.9235,  7.7883,  8.2426,
#          8.3304, 10.4975, 10.8797, 11.3158, 11.4658, 11.4990]],
#       device='cuda:0')

####################################################

# Example with K = 1 and REDQ-Blau
# thetas = [[ 1.4350, -1.4982]]

# actions = [
#     [-0.0781, -0.0491],
#     [ 0.2842,  0.2789],
#     [ 0.3628, -0.2019],
#     [ 0.1327, -0.3327],
#     [ 0.4216, -0.1134],
#     [-0.2576,  0.5183],
#     [-0.5621, -0.1794],
#     [-0.0786, -0.5541],
#     [ 0.5336,  0.0614],
#     [-0.5181, -0.2729],
#     [ 0.4861, -0.5012],
#     [ 0.3675, -0.4531],
#     [ 0.4296, -0.5014],
#     [ 0.4024, -0.3532],
#     [ 0.4879, -0.6102],
#     [-0.5533,  0.4741],
#     [-0.2527,  0.4677],
#     [ 0.5797, -0.6073],
#     [ 0.3282, -0.2884],
#     [ 0.1751, -0.3581],
#     [ 0.1279, -0.4223],
#     [ 0.4799, -0.3795],
#     [ 0.4934, -0.3086],
#     [ 0.1113, -0.1984],
#     [ 0.3268, -0.4066],
#     [-0.0583,  0.3234],
#     [ 0.2991,  0.1691],
#     [ 0.1501, -0.0922],
#     [-0.7551, -0.4198],
#     [ 0.3153, -0.2821]
# ]

# Reward in sPCE
#torch.tensor([[1.2168, 1.8782, 3.7437, 4.3356, 5.0065, 5.0103, 5.0733, 5.0950, 5.2498,
#         5.2885, 5.6458, 6.0531, 5.8517, 6.6363, 7.1932, 7.2368, 7.2332, 7.3497,
#         8.1514, 8.2016, 8.2149, 8.2899, 8.2741, 8.2852, 8.6271, 8.6297, 8.6331,
#         8.6379, 8.6368, 8.5398]], device='cuda:0')

####################################################

# Example with K = 1 and REDQ-Blau
# thetas = [[-1.4033,  1.8134]]

# actions = [
#     [ 0.0619,  0.0410],
#     [-0.2723, -0.3064],
#     [-0.3346,  0.2022],
#     [-0.0421,  0.4235],
#     [ 0.2087, -0.3828],
#     [ 0.4161,  0.2384],
#     [-0.4467,  0.1148],
#     [-0.2411,  0.5426],
#     [-0.2899,  0.4508],
#     [-0.2452,  0.5105],
#     [-0.2622,  0.4699],
#     [-0.2022,  0.4378],
#     [-0.3410,  0.4136],
#     [-0.5244, -0.1053],
#     [ 0.3183, -0.5616],
#     [ 0.0264, -0.5033],
#     [-0.0973,  0.2812],
#     [ 0.4743,  0.1304],
#     [-0.3926,  0.6752],
#     [-0.4656,  0.5166],
#     [-0.3696,  0.5098],
#     [-0.2965,  0.7137],
#     [-0.4344,  0.6111],
#     [-0.1938,  0.3867],
#     [-0.1570,  0.5367],
#     [-0.1510,  0.5598],
#     [-0.1987,  0.5881],
#     [-0.4295,  0.5496],
#     [-0.2245,  0.2088],
#     [-0.4076,  0.3019]
# ]

# Reward in sPCE
#torch.tensor([[1.3280, 1.5554, 3.1007, 4.3434, 4.3448, 4.3842, 5.0775, 5.6059, 6.5310,
#         6.9091, 7.1236, 7.6223, 8.7162, 8.7721, 8.7615, 8.7417, 8.6876, 8.7128,
#         8.7171, 8.9346, 9.0116, 9.0946, 9.0773, 9.0696, 9.0887, 9.0927, 9.0831,
#         9.1309, 9.1451, 9.2906]], device='cuda:0')

####################################################

# Example with K = 1 and REDQ-Blau
# thetas = [[-0.1029,  1.6810]]

# actions = [
#     [ 0.0190, -0.0991],
#     [ 0.2229,  0.2593],
#     [-0.0646,  0.3495],
#     [-0.1768,  0.2205],
#     [ 0.4308, -0.1251],
#     [-0.4559, -0.1154],
#     [ 0.0589,  0.5181],
#     [ 0.2203,  0.5079],
#     [-0.0651,  0.6189],
#     [ 0.2674,  0.4740],
#     [-0.0739, -0.6725],
#     [-0.6699,  0.2714],
#     [-0.2958,  0.5423],
#     [-0.6115,  0.2271],
#     [ 0.7587, -0.2241],
#     [ 0.5404,  0.4344],
#     [-0.1605,  0.6910],
#     [ 0.0425,  0.6151],
#     [-0.0806,  0.4425],
#     [ 0.2617,  0.7117],
#     [ 0.0273, -0.7955],
#     [ 0.5827, -0.3670],
#     [-0.7243, -0.1339],
#     [-0.2533,  0.5801],
#     [-0.1852,  0.7380],
#     [-0.2434,  0.6501],
#     [-0.5309,  0.5893],
#     [ 0.3638,  0.5844],
#     [ 0.2420,  0.2156],
#     [ 0.1958, -0.4651]
# ]

# Reward in sPCE
#torch.tensor([[1.0161, 2.3201, 2.1261, 2.3423, 2.3479, 2.1483, 2.9695, 2.2009, 1.5085,
#         4.1328, 4.1516, 4.2832, 4.5904, 4.6367, 4.6449, 4.6670, 4.8249, 4.8801,
#         5.9245, 5.7651, 5.7629, 5.7984, 5.7789, 5.9147, 6.3286, 5.3920, 5.2195,
#         4.7314, 5.1283, 5.0871]], device='cuda:0')

####################################################

# Example with K = 3 and REDQ-Blau
# thetas = [[-0.1029,  1.6810], [-0.2708, -0.5432], [-0.3521,  0.3413]]

# actions = [
#     [-0.0734,  0.0907],
#     [ 0.2361,  0.0202],
#     [-0.1988, -0.1856],
#     [ 0.0608, -0.4328],
#     [-0.2692,  0.1873],
#     [-0.2795,  0.0638],
#     [-0.3093,  0.0691],
#     [-0.1333,  0.2729],
#     [-0.2174,  0.2716],
#     [-0.0930,  0.2254],
#     [ 0.0343,  0.1402],
#     [-0.2438,  0.1996],
#     [-0.0947,  0.2639],
#     [-0.0299, -0.0171],
#     [-0.2145,  0.1620],
#     [-0.3189,  0.0833],
#     [-0.0450,  0.0204],
#     [-0.2242,  0.1302],
#     [-0.3543,  0.1077],
#     [-0.4299, -0.0003],
#     [-0.1089,  0.0387],
#     [-0.1371, -0.0229],
#     [-0.1020, -0.0754],
#     [ 0.0439, -0.0241],
#     [-0.1035,  0.0295],
#     [ 0.0161,  0.2331],
#     [-0.1182,  0.0655],
#     [ 0.1110,  0.0775],
#     [-0.2550,  0.2800],
#     [-0.1657,  0.1847]
# ]

# Reward in sPCE
#torch.tensor([[3.2799, 3.7761, 1.1443, 1.4008, 1.0548, 1.0327, 1.6552, 2.2237, 2.2024,
#         2.5186, 3.1062, 2.8470, 2.5056, 2.9332, 2.9761, 3.0438, 3.2108, 3.3322,
#         1.5723, 1.6515, 0.7597, 1.9029, 4.8427, 5.4175, 5.2088, 5.7694, 7.5181,
#         7.5043, 7.6710, 7.7462]], device='cuda:0')

####################################################

# Example with K = 3 and REDQ-Blau
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402],
#           [-2.1387, -0.6234]]

# actions = [
#     [-0.1064, -0.0669],
#     [-0.0479,  0.0304],
#     [-0.0275,  0.0672],
#     [-0.0364,  0.0526],
#     [-0.0087, -0.3249],
#     [-0.2584,  0.1448],
#     [-0.3101, -0.0965],
#     [ 0.0495,  0.0927],
#     [ 0.2011,  0.1228],
#     [ 0.1809,  0.0976],
#     [-0.0470,  0.4036],
#     [ 0.3714, -0.0250],
#     [-0.5207,  0.2026],
#     [-0.3447,  0.1070],
#     [-0.3196,  0.2849],
#     [-0.3269,  0.2862],
#     [-0.2635,  0.2205],
#     [-0.1666,  0.3107],
#     [-0.2028,  0.2820],
#     [-0.2985,  0.1012],
#     [-0.3377,  0.1967],
#     [-0.2703,  0.1063],
#     [-0.2425,  0.1343],
#     [-0.4016,  0.2671],
#     [-0.3145,  0.3881],
#     [-0.4564,  0.4493],
#     [-0.3600,  0.3071],
#     [-0.2907,  0.5645],
#     [-0.2912,  0.4831],
#     [-0.1409,  0.5092]
# ]

# Reward in sPCE
#torch.tensor([[ 0.5548,  3.2761,  4.3855,  5.4385,  6.6415,  7.0744,  6.2923,  5.8955,
#          5.5029,  5.4083,  6.2424,  8.0696,  9.2189,  9.4573, 10.3200, 10.9195,
#         11.0302, 11.3382, 11.4871, 11.5456, 11.6170, 11.7200, 11.0758, 10.3293,
#         11.9329, 12.8696, 13.3513, 13.3723, 13.2470, 13.3595]],
#       device='cuda:0')

####################################################

# Example with K = 3 and REDQ-Blau
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784],
#           [-2.6479, -0.0233]]

# actions = [
#     [-0.0522,  0.0929],
#     [ 0.1927, -0.2249],
#     [ 0.2424,  0.0077],
#     [-0.0119, -0.2509],
#     [-0.1370, -0.2038],
#     [-0.1193, -0.1987],
#     [-0.2315, -0.1085],
#     [ 0.3829,  0.1481],
#     [-0.0337, -0.3604],
#     [-0.0140, -0.3642],
#     [ 0.1098,  0.2627],
#     [-0.3593, -0.3112],
#     [-0.2983, -0.3377],
#     [-0.2910, -0.3191],
#     [-0.3091, -0.3022],
#     [-0.2550, -0.2696],
#     [-0.2582, -0.2320],
#     [-0.2250, -0.2597],
#     [ 0.2186, -0.2888],
#     [ 0.3377, -0.2406],
#     [ 0.4028, -0.2094],
#     [ 0.3218, -0.1831],
#     [ 0.3955, -0.2060],
#     [ 0.4122, -0.0587],
#     [ 0.3222, -0.1864],
#     [ 0.3154, -0.1627],
#     [ 0.2325, -0.3011],
#     [ 0.1887, -0.4082],
#     [-0.0578, -0.2650],
#     [ 0.0367, -0.3736]
# ]

# Reward in sPCE
#torch.tensor([[ 2.1506,  2.9861,  2.8815,  3.7866,  4.4490,  4.3914,  5.1564,  6.3949,
#          4.6211,  5.9993,  6.3284,  6.2990,  6.6583,  6.9177,  7.0705,  6.7622,
#          7.1107,  7.9996,  9.3226, 10.6339, 11.0304, 11.0565, 11.2767, 11.2737,
#         11.3411, 11.4202, 10.8603, 11.5267, 12.0206, 11.8552]],
#       device='cuda:0')

####################################################

# Example with K = 4 and REDQ-Blau
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784],
#           [-2.6479, -0.0233],
#           [-1.3761,  1.1551]]

# actions = [
#     [ 0.0476, -0.1262],
#     [ 0.0233,  0.1802],
#     [-0.2431,  0.0066],
#     [-0.1611,  0.2630],
#     [-0.2363, -0.2720],
#     [-0.1491, -0.2243],
#     [-0.2053,  0.3526],
#     [-0.2269,  0.2993],
#     [-0.1423, -0.2340],
#     [-0.2230,  0.2320],
#     [-0.1679, -0.2514],
#     [-0.2194,  0.3035],
#     [-0.1495, -0.2579],
#     [-0.1919,  0.2666],
#     [-0.0890, -0.2688],
#     [-0.1647, -0.2169],
#     [-0.1829,  0.2695],
#     [-0.0623, -0.2739],
#     [-0.1509, -0.1854],
#     [-0.1850,  0.2188],
#     [-0.1484, -0.1751],
#     [-0.2244,  0.2487],
#     [-0.2592,  0.1229],
#     [-0.0432, -0.2211],
#     [-0.1508, -0.1722],
#     [-0.2154, -0.0910],
#     [-0.1026,  0.3136],
#     [-0.3176,  0.2466],
#     [-0.1450,  0.3540],
#     [-0.2167,  0.2551]
# ]

# Reward in sPCE
#torch.tensor([[ 1.9042,  3.3457,  4.1309,  4.9221,  6.3112,  6.6852,  7.0988,  7.6359,
#          7.7357,  8.0891,  7.8482,  7.9668,  7.5207,  7.5868,  7.4880,  7.8988,
#          7.9445,  7.4921,  7.6753,  7.7891,  8.0845,  8.3065,  8.1515,  8.3272,
#          8.4425,  8.7560, 10.3401, 10.8896, 10.5642, 10.6205]],
#       device='cuda:0')

####################################################

# Example with K = 4 and REDQ-Blau
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402],
#           [-2.1387, -0.6234],
#           [-0.4541, -0.6023]]

# actions = [
#     [-0.0052, -0.3285],
#     [-0.0257, -0.0297],
#     [0.0635, -0.0100],
#     [-0.1117, -0.1477],
#     [0.2854, 0.0608],
#     [0.0025, 0.3238],
#     [-0.1591, 0.2671],
#     [-0.1665, 0.1694],
#     [-0.0664, 0.1154],
#     [-0.0253, 0.1390],
#     [-0.0940, 0.1113],
#     [0.0273, 0.0373],
#     [-0.1118, 0.0764],
#     [-0.0356, 0.0437],
#     [-0.0656, 0.0501],
#     [-0.1610, 0.0849],
#     [-0.1522, 0.0907],
#     [-0.0602, -0.2558],
#     [-0.1029, 0.1663],
#     [-0.3028, -0.0635],
#     [0.0570, -0.0062],
#     [-0.1088, -0.0114],
#     [-0.1928, -0.0869],
#     [-0.2098, -0.2143],
#     [-0.1766, -0.1057],
#     [-0.2410, -0.0363],
#     [0.0275, 0.0317],
#     [-0.1845, 0.1146],
#     [-0.1380, -0.0063],
#     [-0.1086, -0.1101]
# ]

# Reward in sPCE
#torch.tensor([[ 0.5020,  0.6512,  1.0713,  7.8589,  9.3896,  9.9376, 10.3714, 10.6999,
#         11.9919, 12.7775, 12.9277, 12.9384, 13.0918, 13.5875, 13.4773, 13.4900,
#         13.4742, 13.4726, 13.4532, 13.4583, 13.4214, 13.5510, 13.5724, 13.6422,
#         13.6567, 13.6579, 13.6764, 13.6357, 13.5925, 13.5745]],
#       device='cuda:0')

####################################################

# Example with K = 4 and REDQ-Blau
# thetas = [[-0.1029,  1.6810],
#           [-0.2708, -0.5432],
#           [-0.3521,  0.3413],
#           [-1.6456,  1.5698]]

# actions = [
#     [-0.0592, 0.0700],
#     [-0.0257, 0.0569],
#     [-0.0340, 0.0626],
#     [-0.0550, 0.0630],
#     [0.1506, -0.2180],
#     [-0.1133, -0.0207],
#     [-0.2088, -0.0237],
#     [-0.2204, -0.1643],
#     [-0.0915, 0.0077],
#     [-0.1274, 0.0396],
#     [-0.0864, 0.0553],
#     [-0.0920, 0.0672],
#     [-0.0179, 0.1284],
#     [-0.0592, 0.0841],
#     [-0.1391, 0.1125],
#     [-0.0202, 0.0246],
#     [-0.0190, 0.1052],
#     [-0.1401, 0.1215],
#     [-0.0832, 0.0725],
#     [-0.1182, 0.0642],
#     [-0.0263, 0.0448],
#     [-0.1154, 0.0343],
#     [-0.0130, 0.0211],
#     [0.0321, 0.0668],
#     [-0.0278, 0.1010],
#     [-0.0246, 0.1020],
#     [-0.0214, -0.0164],
#     [-0.1431, 0.0876],
#     [-0.1190, -0.0349],
#     [-0.0196, 0.1342]
# ]

# Reward in sPCE
#torch.tensor([[3.0476, 3.2366, 3.8386, 3.9551, 4.3468, 4.1072, 3.5319, 3.8592, 2.2887,
#         2.9800, 4.1197, 4.6433, 3.6890, 4.1460, 4.9426, 6.3652, 5.3327, 5.0962,
#         5.1823, 5.5401, 5.5864, 5.5061, 5.5452, 5.0674, 6.5123, 7.0361, 7.2698,
#         7.2292, 7.5604, 7.9505]], device='cuda:0')

####################################################

# Example with K = 5 and REDQ-Blau
# thetas = [[-0.1029,  1.6810],
#           [-0.2708, -0.5432],
#           [-0.3521,  0.3413],
#           [-1.6456,  1.5698],
#           [-0.2490, -0.4059]]

# actions = [
#     [-0.0516, 0.0675],
#     [-0.0371, 0.0524],
#     [-0.0137, 0.0257],
#     [0.0003, -0.0150],
#     [0.1067, -0.1898],
#     [0.0388, -0.0496],
#     [-0.0056, -0.0023],
#     [0.0316, 0.0121],
#     [0.0042, 0.0196],
#     [0.1650, -0.0325],
#     [0.2357, 0.0384],
#     [0.1169, 0.1652],
#     [0.0287, -0.0743],
#     [-0.0684, 0.0467],
#     [-0.0864, 0.0229],
#     [-0.0300, -0.0387],
#     [-0.0224, -0.0254],
#     [0.0329, -0.0200],
#     [-0.0560, -0.0201],
#     [0.0261, -0.0329],
#     [0.0701, 0.0214],
#     [-0.0189, 0.0320],
#     [0.2120, 0.1148],
#     [0.0382, -0.0535],
#     [-0.1519, 0.0719],
#     [-0.1908, -0.0382],
#     [-0.1057, 0.0957],
#     [-0.1617, 0.0952],
#     [-0.1429, 0.0939],
#     [-0.1749, 0.1241]
# ]

# Reward in sPCE
#torch.tensor([[2.2888, 3.4653, 3.4376, 3.2949, 3.6784, 2.7019, 3.1394, 3.8763, 4.7709,
#         3.2846, 3.6458, 4.4193, 5.6576, 5.2193, 5.2325, 5.3502, 5.9939, 6.3764,
#         6.6697, 6.7426, 6.5454, 7.2365, 7.7754, 8.0284, 8.1140, 8.5354, 9.7725,
#         9.7926, 9.5149, 9.9817]], device='cuda:0')

####################################################

# Example with K = 5 and REDQ-Blau
# thetas = [[ 1.4350, -1.4982],
#           [-0.8099, -1.1784],
#           [-2.6479, -0.0233],
#           [-1.3761,  1.1551],
#           [-1.0914, -0.3823]]

# actions = [
#     [0.0144, -0.0237],
#     [-0.1343, 0.1455],
#     [0.2272, -0.0222],
#     [-0.0737, -0.1725],
#     [-0.1661, -0.1149],
#     [-0.0809, -0.1138],
#     [0.1423, 0.3560],
#     [-0.2255, -0.0760],
#     [0.1016, -0.1379],
#     [-0.2189, -0.0483],
#     [0.2450, 0.2029],
#     [-0.2514, 0.2162],
#     [-0.2152, 0.0417],
#     [-0.1812, 0.2634],
#     [-0.0717, 0.0666],
#     [-0.0402, -0.0047],
#     [-0.1183, -0.0712],
#     [-0.0243, -0.0036],
#     [-0.0632, -0.0594],
#     [-0.1126, 0.1557],
#     [-0.2592, 0.0992],
#     [-0.2744, -0.0787],
#     [-0.2185, 0.3236],
#     [-0.1388, 0.3729],
#     [-0.1840, 0.2996],
#     [-0.2687, 0.1743],
#     [-0.3625, 0.0476],
#     [-0.3070, 0.1816],
#     [-0.1452, 0.2041],
#     [-0.3464, 0.1213]
# ]

# Reward in sPCE
#torch.tensor([[ 1.6495,  2.0241,  3.8460,  3.9589,  4.8259,  5.2610,  6.0096,  6.4022,
#          6.8524,  6.4329,  6.5772,  6.9221,  7.1712,  8.1269,  8.1175,  8.1474,
#          8.2178,  8.2484,  8.2865,  8.1663,  8.3925, 10.3790, 10.9935, 10.3993,
#         10.1365, 10.5298, 10.7841, 10.6939, 10.6477, 10.4280]],
#       device='cuda:0')

####################################################

# Example with K = 5 and REDQ-Blau
# thetas = [[-1.4033,  1.8134],
#           [-0.2177,  0.2402],
#           [-2.1387, -0.6234],
#           [-0.4541, -0.6023],
#           [ 0.3330, -2.1362]]

# actions = [
#     [-0.1420, -0.1265],
#     [0.0761, -0.0781],
#     [-0.1008, -0.0224],
#     [0.1235, 0.2314],
#     [-0.1098, 0.1595],
#     [-0.0784, 0.0717],
#     [-0.1090, -0.1090],
#     [-0.1174, 0.0329],
#     [-0.0810, -0.0057],
#     [-0.0947, 0.0762],
#     [-0.0121, -0.1347],
#     [-0.1087, -0.0900],
#     [-0.1053, 0.0634],
#     [-0.1265, -0.0196],
#     [-0.0431, -0.1703],
#     [-0.0839, 0.0019],
#     [-0.1672, -0.0675],
#     [-0.0281, -0.0818],
#     [-0.0683, -0.0249],
#     [-0.1094, -0.0132],
#     [-0.1381, -0.0113],
#     [-0.1198, 0.0673],
#     [-0.1473, 0.0434],
#     [-0.0159, 0.0753],
#     [-0.1307, 0.1698],
#     [-0.0693, -0.0288],
#     [-0.0445, -0.0651],
#     [-0.0910, 0.0127],
#     [-0.1008, -0.0666],
#     [-0.0315, 0.0128]
# ]

# Reward in sPCE
#torch.tensor([[ 2.6468,  3.4340,  3.9046,  4.7211,  5.5870,  8.2436,  9.4339,  9.6619,
#          9.7363,  9.9024,  9.8295,  9.5894,  9.8338, 10.3350, 10.6241, 10.7092,
#         10.8548, 10.7647, 10.4156, 10.3507, 10.4112, 11.2077, 11.2445, 10.8034,
#         12.2201, 12.2764, 12.2338, 12.2266, 12.5533, 12.7563]],
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
plt.savefig(f"location_finding_redq_blau_k_{len(thetas)}.pdf", transparent=True, bbox_inches="tight")
#plt.show()
