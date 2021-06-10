import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy

from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp, HawkesEM)
from tick.base import TimeFunction
from tick.plot import plot_hawkes_kernels
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

seed = 1111
np.random.seed(seed)

def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))/(2 * np.pi * np.power(sig, 2.)) ** 0.5

def kernel(dt, means, sigs, alpha):
    k = gaussian(dt[:, None], means[None, :], sigs[None, :]).dot(alpha)
    return k

def simulHawkes(lamb0, alpha, means, sigs, run_time=1000):
    maxdt = max(means)+3*max(sigs)
    nbClasses = len(alpha)
    # Definition kernels
    emptyKer = HawkesKernelTimeFunc(TimeFunction(([0, 1], [0, 0]), inter_mode=TimeFunction.InterConstRight))
    kernels = [[copy(emptyKer) for _ in range(nbClasses)] for _ in range(nbClasses)]

    for c in range(nbClasses):
        for c2 in range(nbClasses):
            if c!=c2: continue  # Univariate Hawkes process
            t_values = np.linspace(0, maxdt, 100)
            y_values = kernel(t_values, means, sigs, alpha[c,c2])

            #plt.plot(t_values, y_values)
            #plt.show()

            tf = TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight, dt=maxdt/100)
            #kernels.append(HawkesKernelTimeFunc(tf))
            kernels[c][c2] = HawkesKernelTimeFunc(tf)

    baseline = np.array([lamb0 for _ in range(nbClasses)])

    hawkes = SimuHawkes(baseline=baseline, end_time=run_time, verbose=False, seed=int(np.random.random()*10000))

    for c in range(nbClasses):
        for c2 in range(nbClasses):
            if c!=c2: continue  # Univariate Hawkes process
            hawkes.set_kernel(c, c2, kernels[c][c2])

    hawkes.simulate()

    events = []
    for c, _ in enumerate(baseline):
        events.append([c,0])
    for c in range(len(hawkes.timestamps)):
        for t in hawkes.timestamps[c]:
            events.append([c, t])
    events = np.array(events)

    return events, hawkes

def simulTxt(events, voc_per_class, nbClasses, overlap_voc, words_per_obs):
    # Generate text
    # Perfectly separated text content
    voc_clusters = [np.array(list(range(int(voc_per_class)))) + c*voc_per_class for c in range(nbClasses)]

    # Overlap
    for c in range(nbClasses):
        voc_clusters[c] -= int(c*voc_per_class*overlap_voc)

    # Associate a fraction of vocabulary to each observation
    arrtxt = []
    for e in events:
        c_text = int(e[1])
        arrtxt.append(np.random.choice(voc_clusters[c_text], size=words_per_obs))

    return arrtxt

def compute_kernel_alltimes(events, means, sigs, alpha, res=1000):
    ranget = np.linspace(0, np.max(events[:, -1]), res)
    tabvals = [[] for _ in range(nbClasses)]
    maxdt = max(means)+3*max(sigs)
    for t in ranget:
        ev = events[events[:, -1]>t-maxdt]
        ev = ev[ev[:, -1]<t]
        for c in range(nbClasses):
            eventsprec = ev[ev[:, 0] == c]  # 0 bc has to be temporal clusters
            val = kernel(t - eventsprec[:, -1], means, sigs, alpha[c,c]).sum()
            tabvals[c].append(val)
    tabvals = np.array(tabvals)

    return ranget, tabvals[0], tabvals[1]

def compute_overlap_temp(x, y1, y2):
    area1 = np.trapz(y1, x=x)
    area2 = np.trapz(y2, x=x)
    areaInter = np.trapz(np.min([y1,y2], axis=0), x=x)

    overlap = 2*areaInter/(area1+area2)

    return overlap

def make_overlap_temp(events, alpha, overlap_temp, params_resimul):
    ol_temp = -1000
    eps = 0.05
    dt = 10
    res = 1000
    while not (ol_temp>overlap_temp-eps and ol_temp<overlap_temp+eps):
        maxt = np.max(events[:, -1])
        i=0
        while not (ol_temp>overlap_temp-eps and ol_temp<overlap_temp+eps) and nbClasses == 2 and i*dt<maxt:
            events[events[:, 0]==0, 1] += dt
            t, kernel1, kernel2 = compute_kernel_alltimes(events, means, sigs, alpha, res=res)
            ol_temp = compute_overlap_temp(t, kernel1, kernel2)
            i+=1
            #print(i*dt, ol_temp, maxt)
            if ol_temp<overlap_temp:
                break

        if not (ol_temp>overlap_temp-eps and ol_temp<overlap_temp+eps):
            events, hawkes = simulHawkes(*params_resimul)

    return events

def save(folder, name, events, arrtxt, lamb0, means, sigs, alpha):
    events = np.insert(events, 3, np.array(list(range(len(events)))), axis=1)
    events = np.array(list(sorted(events, key= lambda x: x[2])))
    with open(folder+name+"_events.txt", "w+") as f:
        for i, e in enumerate(events):
            content = ",".join(map(str, list(arrtxt[int(e[3])])))
            txt = str(e[0])+"\t"+str(e[1])+"\t"+str(e[2])+"\t"+content+"\n"
            f.write(txt)

    with open(folder+name+"_lamb0.txt", "w+") as f:
        f.write(str(lamb0))
    np.savetxt(folder+name+"_means.txt", means)
    np.savetxt(folder+name+"_sigs.txt", sigs)
    np.save(folder+name+"_alpha", alpha)

def plotProcess(events, means, sigs, alpha, whichclus=0):
    colors = ["r", "b", "y", "g", "orange", "cyan","purple"]
    maxdt = max(means)+3*max(sigs)
    nbClasses = len(alpha)
    rangedt = np.linspace(0, maxdt, 100)
    for e in events:
        c = int(e[whichclus])
        t = e[-1]
        #plt.plot(t, -1/10-c/10, "o", c=colors[c], markersize=4)
        #plt.plot(t+rangedt, kernel(rangedt, means, sigs, alpha[c]), colors[c], alpha=0.1)

    ranget = np.linspace(0, np.max(events[:, -1]), 10000)
    tabvals = [[] for _ in range(nbClasses)]
    for t in ranget:
        ev = events[events[:, -1]>t-maxdt]
        ev = ev[ev[:, -1]<t]
        for c in range(nbClasses):
            eventsprec = ev[ev[:, 0] == c]  # 0 bc has to be temporal clusters
            val = kernel(t - eventsprec[:, -1], means, sigs, alpha[c,c]).sum()
            tabvals[c].append(val)
    tabvals = np.array(tabvals)

    for c in range(nbClasses):
        plt.plot(ranget, lamb0+tabvals[c], "-", c=colors[c])

    plt.show()



def generate(params):
    nbClasses, run_time, voc_per_class, overlap_voc, overlap_temp, voc_per_class, perc_rand, words_per_obs, run, lamb0, means, sigs, folder = params
    maxdt = max(means)+3*max(sigs)
    alpha = np.zeros((nbClasses, nbClasses, len(means)))
    for c in range(nbClasses):
        a = np.random.random((len(means)))
        alpha[c,c]=a/np.sum(a)
        for c2 in range(nbClasses):
            if c==c2: continue
            a = np.random.random((len(means)))
            alpha[c,c2] = 0
    alpha = np.array(alpha)

    # Get timestamps and temporal clusters
    events = []
    events, hawkes = simulHawkes(lamb0, alpha, means, sigs, run_time=run_time)
    print(len(events), "events")
    dofit = False
    if dofit:
        em = HawkesEM(15, kernel_size=30, n_threads=8, verbose=False, tol=1e-3)
        em.fit(hawkes.timestamps)
        fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
        plt.show()

    # Get the wanted temporal overlap
    if overlap_temp >=0 and nbClasses==2:
        params_resimul=(lamb0, alpha, means, sigs, run_time)
        events = make_overlap_temp(events, alpha, overlap_temp, params_resimul)

    # Initialize textual clusters and shuffle nb_rand of them
    events = np.insert(events, 0, events[:, 0], axis=1)
    nb_rand = int(perc_rand*len(events))
    events[np.random.randint(0, len(events), nb_rand), 1] = np.random.randint(0, nbClasses, nb_rand)

    # Generate text associated with textual clusters
    arrtxt = simulTxt(events, voc_per_class, nbClasses, overlap_voc, words_per_obs)

    # Plot the process (univariate only e.g. diagonal of alpha)
    #print(len(events))
    #plotProcess(events, means, sigs, alpha, whichclus=1)
    #pause()

    name = f"Obs_nbclasses={nbClasses}_lg={run_time}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_run={run}"
    save(folder, name, events, arrtxt, lamb0, means, sigs, alpha)

nbClasses = 2
run_time = 1500
XP = "Overlap"

overlap_voc = 0.  # Proportion of voc in common between a clusters and its direct neighbours
overlap_temp = 0.  # Overlap between the kernels of the simulating process

voc_per_class = 1000  # Number of words available for each cluster
perc_rand = 0.  # Percentage of events to which assign random textual cluster
words_per_obs = 100

run = 0

lamb0 = 0.05
means = np.array([3, 7, 11])
sigs = np.array([0.5, 0.5, 0.5])
folder = "data/Synth/"
np.random.seed(1564)
#params = (nbClasses, run_time, voc_per_class, overlap_voc, overlap_temp, voc_per_class, perc_rand, words_per_obs, run, lamb0, means, sigs, folder)
params = (2, 60, voc_per_class, 0.3, 0.4, voc_per_class, perc_rand, words_per_obs, 0, lamb0, means, sigs, folder)
generate(params)
pause()
nbRuns = 10
if XP == "Decorr":
    for perc_rand in np.array(list(range(11)))/10:
        for run in range(nbRuns):
            params = (nbClasses, run_time, voc_per_class, overlap_voc, overlap_temp, voc_per_class, perc_rand, words_per_obs, run, lamb0, means, sigs, folder)
            print(f"{nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp} - perc_rand={perc_rand} - run={run}")
            generate(params)
elif XP == "Overlap":
    np.random.seed(14776)
    for overlap_voc in [0., 0.3, 0.5, 0.7, 0.9]:
        for overlap_temp in [0., 0.3, 0.5, 0.7]:
            for run in range(nbRuns):
                params = (nbClasses, run_time, voc_per_class, overlap_voc, overlap_temp, voc_per_class, perc_rand, words_per_obs, run, lamb0, means, sigs, folder)
                print(f"{nbClasses} classes - OL_text={overlap_voc} - OL_temp={overlap_temp} - perc_rand={perc_rand} - run={run}")
                generate(params)





