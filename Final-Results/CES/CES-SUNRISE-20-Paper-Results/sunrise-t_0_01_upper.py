from torch import tensor, cat, stack, mean
import numpy as np

t1 = 

t2 = 

t3 = 

t4 = 

t5 = 

t6 = 

t7 = 

t8 = 

t9 = 

t10 = 

tensors = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]

last_elements = [tensor[:, -1] for tensor in tensors]

all_last_elements = cat(last_elements)

mean_last_elements = all_last_elements.mean().item()
se_last_elements = all_last_elements.std().item() / np.sqrt(all_last_elements.numel())

print("mean:", mean_last_elements, "\nse:", se_last_elements)

stacked_tensors = stack(tensors)

mean_values = mean(stacked_tensors, dim=0)

mean_vals = mean(mean_values, dim=0)

print(mean_values.shape)
print(mean_values)

print(mean_vals.shape)
print(mean_vals)