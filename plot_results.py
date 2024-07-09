import numpy as np
import matplotlib.pyplot as plt

def compute_central_tendency_and_error(id_central, id_error, sample):

    try:
        id_error = int(id_error)
    except:
        pass

    if id_central == 'mean':
        central = np.nanmean(sample, axis=0)
    elif id_central == 'median':
        central = np.nanmedian(sample, axis=0)
    else:
        raise NotImplementedError

    if isinstance(id_error, int):
        low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=0)
        high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=0)
    elif id_error == 'std':
        low = central - np.nanstd(sample, axis=0)
        high = central + np.nanstd(sample, axis=0)
    elif id_error == 'sem':
        low = central - np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0])
        high = central + np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0])
    else:
        raise NotImplementedError

    return central, low, high

#plt.rcParams['axes.labelsize'] = 22
#plt.rcParams['axes.titlesize'] = 14
#plt.rcParams['xtick.labelsize'] = 12
#plt.rcParams['ytick.labelsize'] = 12
#plt.rcParams['legend.fontsize'] = 15
#plt.rcParams['figure.titlesize'] = 36

fig_dpi = 100
width, height = (10, 6)
#titlesize = 20

data = np.load("../sbr300000_results.npz")
step = 50
L = 1e5
err_bar = "se"
title_template = "SBR-300000: Location Finding"
file_template = "sbr300000_new_central_tend"

typ = "rmeans" if err_bar in ["std", "se"] else "rmedians"
means = data[typ].astype(np.float64).reshape((-1,))
# Makes the 10 different seeds stack together and calculate overall mean of these 10 together
means = np.array_split(means, 10)
means_ = np.stack(means)

central, low, high = compute_central_tendency_and_error("mean", "sem", means_)
print("low, high", low, high)

means = np.mean(means_, axis=0)
print("Number of means:", means.size)
#smoothed_means = np.asarray([means[i:i+step].mean() for i in range(means.size - step)])
#stds_means = np.std(means_, axis=0)
#print("Number of stds:", len(stds_means))
#print(stds_means)

plt.figure(figsize=(width, height), dpi=fig_dpi)
xlim, ylim = [0, 20001], [0, 12]
plt.plot(np.arange(0, means.size), central, color = "red", label = "Mean sPCE")
if err_bar in ["std", "se"]:
	stds = data['rstds'].astype(np.float64).reshape((-1,))
	stds = np.array_split(stds, 10)
	stds = np.stack(stds)
	stds = np.mean(stds, axis=0)
	#smoothed_stds = np.asarray([stds[i:i+step].mean() for i in range(stds.size - step)])
	if err_bar == "se":
		stds /= np.sqrt(10)
	upper_bound = means + stds
	lower_bound = means - stds
elif err_bar == "iqr":
	uqs = data['ruqs'].astype(np.float64).reshape((-1,))
	uqs = np.array_split(uqs, 10)
	uqs = np.stack(uqs)
	uqs = np.mean(uqs, axis=0)
	#smoothed_uqs = np.asarray([uqs[i:i+step].mean() for i in range(uqs.size - step)])
	lqs = data['rlqs'].astype(np.float64).reshape((-1,))
	lqs = np.array_split(lqs, 10)
	lqs = np.stack(lqs)
	lqs = np.mean(lqs, axis=0)
	#smoothed_lqs = np.asarray([lqs[i:i+step].mean() for i in range(lqs.size - step)])
	upper_bound, lower_bound = uqs, lqs
print("lower_bound, upper_bound", lower_bound, upper_bound)
plt.fill_between(
	np.arange(0, means.size), low, high, alpha=0.5, label = "Standard Error"
)
plt.hlines(np.log(L+1), xmin=xlim[0], xmax=xlim[1], color="black", linestyles='dashed', label = "log(100000+1)")
print(means[-10:])
plt.xlabel("Iterations")
plt.ylabel("sPCE")
plt.title(f"{title_template}")
plt.yticks([0, 2, 4, 6, 8, 10, 12])
#plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"training_{file_template}.pdf")