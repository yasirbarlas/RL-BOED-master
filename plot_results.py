import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

#folders = ["../source_sunrise10_results.npz", "../source_sunrise20_results.npz"]
#labels = ["SUNRISE-10", "SUNRISE-20"]
#title_template = "Location Finding: SUNRISE"
#file_template = "sunrise_new_central_tend"

#folders = ["../source_sbr300000_results.npz", "../source_sbr430000_results.npz"]
#labels = ["SBR-300000", "SBR-430000"]
#title_template = "Location Finding: Scaled-by-Resetting"
#file_template = "sbr_new_central_tend"

#folders = ["../source_droq0.01_results.npz", "../source_droq0.1_results.npz"]
#labels = ["DroQ-0.01", "DroQ-0.1"]
#title_template = "Location Finding: Dropout Q-Functions"
#file_template = "droq_new_central_tend"

folders = ["../source_redq_results.npz", "../source_discount0.99_results.npz", "../source_ensemble10_results.npz", "../source_tau0.01_results.npz"]
labels = ["REDQ-Blau", "REDQ-Disc-0.99", "REDQ-Ens-10", "REDQ-Tau-0.01"]
title_template = "Location Finding: Randomised Ensembled Double Q-Learning"
file_template = "redq_new_central_tend"

step = 50
L = 1e5
err_bar = "se"
#plt.figure(figsize=(width, height), dpi=fig_dpi)
fig, ax = plt.subplots(figsize=(width, height), dpi=fig_dpi)

inset_position = [0.3, 0.28, 4, 2]  # [x0, y0, width, height]
#inset_position = [0.1, 0.05, 3, 1]  # [x0, y0, width, height]
ax_inset = inset_axes(ax, width=inset_position[2], height=inset_position[3], loc='lower left',
                      bbox_to_anchor=(inset_position[0], inset_position[1], 
                                      inset_position[2], inset_position[3]), 
                      bbox_transform=ax.transAxes)

for ig, folder in enumerate(folders):
    data = np.load(folder)

    typ = "rmeans" if err_bar in ["std", "se"] else "rmedians"
    means = data[typ].astype(np.float64).reshape((-1,))
    # Makes the 10 different seeds stack together and calculate overall mean of these 10 together
    means = np.array_split(means, 10)
    means_ = np.stack(means)

    central, low, high = compute_central_tendency_and_error("mean", "sem", means_)
    #print("low, high", low, high)

    means = np.mean(means_, axis=0)
    print("Number of means:", means.size)
    #smoothed_means = np.asarray([means[i:i+step].mean() for i in range(means.size - step)])
    #stds_means = np.std(means_, axis=0)
    #print("Number of stds:", len(stds_means))
    #print(stds_means)

    xlim, ylim = [0, 20001], [0, 12]
    ax.plot(np.arange(0, means.size), central, label = labels[ig], alpha=0.5)
    ax_inset.plot(np.arange(0, means.size), central, label = labels[ig], alpha=0.5)
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
    #print("lower_bound, upper_bound", lower_bound, upper_bound)
    #plt.fill_between(
    #	np.arange(0, means.size), low, high, alpha=0.5, label = "Standard Error"
    #)

ax.hlines(np.log(L+1), xmin=xlim[0], xmax=xlim[1], color="black", linestyles='dashed', label = "log(100001)")
#print(means[-10:])
ax.set_xlabel("Iterations")
ax.set_ylabel("sPCE")
ax.set_title(f"{title_template}")
ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
ax.grid()
ax.legend(loc="lower right")

ax_inset.set_xlim(15000, 20000)
ax_inset.set_ylim(10, 11.5)
#ax_inset.set_ylim(10, 11)
ax_inset.set_xticks([15000, 17500, 20000])
ax_inset.set_yticks([10, 11.5])
#ax_inset.set_yticks([10, 11])
#ax_inset.grid()

ax_inset.set_title("Zoomed In", fontsize=10)

plt.savefig(f"training_{file_template}.pdf", transparent=True, bbox_inches="tight")