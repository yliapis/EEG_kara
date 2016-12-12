
import mne
import matplotlib.pyplot as plt
import numpy as np


delta = range(0,5)
theta = range(4,9)
alpha = range(8,15)
beta = range(14,40)

delta2 = [0,1,2]
theta2 = [2,3,4]
alpha2 = [4,5,6,7]
beta2 = [7,8,9,10,11,12,13,14,15]

set1 = [delta, theta, alpha, beta]
set2 = [delta2, theta2, alpha2, beta2]


def topoplot(data, kwargs={'vmin':-.75, 'vmax':.75}):
    #assuming use of 62 channels
    global CH_LOCS  
    try:
        CH_LOCS
    except:
        CH_LOCS = _get_locs()
    return mne.viz.plot_topomap(data, CH_LOCS, **kwargs)


def _get_locs():
    # get channel locations
    locs = np.loadtxt("sensor_loc.txt", dtype=float)
    # delete M1, M2
    locs = np.delete(locs, (32,42), axis=0)
    # scale
    locs = -locs/100
    # stereographic projection
    xy = np.zeros((np.size(locs, 0), 2))
    xy[:,0] = locs[:,0]/(1-locs[:,2])
    xy[:,1] = locs[:,1]/(1-locs[:,2])
    #
    return xy


def _get_band(P, frange):
    return P[:,frange].mean(1)


def grid_topoplot(Pdiff, franges=set1, scale=1):
    tick = 1
    fig = plt.figure(figsize=(16,20))
    #
    if scale==1:
        franges=set1
    else:
        franges=set2
    #
    for P in Pdiff:
        for frange in franges:
            ax = plt.subplot(7, 8, tick)
            #ax.imshow(np.random.rand(100,100))
            im, x = topoplot(_get_band(P, frange))#, {'axes': ax, 'vmin':-1.5, 'vmax':1.5})
            if tick%4==1:
                plt.ylabel("subj {}".format(tick//4))
            if tick <=8:
                ax.title.set_text("delta theta alpha beta".split()[(tick-1)%4])
            #if tick == 13*4:
                #im.colorbar()
            tick += 1
    return

