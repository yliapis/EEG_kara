from __future__ import division

import numpy as np
import os
import mne
from scipy import io


def get_epochwave(f, size=None, state_vect=False):
    label_dict = {'clearing_inds': 1, 'thinking_inds': 2, 'speaking_inds':3}
    ind_dict = io.loadmat(f)
    # format indices
    for key in label_dict.keys():
        ind_dict[key] = np.array(ind_dict[key].tolist()).squeeze()
    #
    if not size:
        size = np.max(np.concatenate([ind_dict[key].flatten() for key in label_dict.keys()]))
        print size
    Y = np.zeros(size)
    if state_vect:
        Y2 = np.zeros((size, 3))
    for key,val in label_dict.items():
        label = label_dict[key]
        for ind0,indf in ind_dict[key]:
            Y[ind0-1:indf] = label
            if state_vect:
                Y2[ind0-1:indf,label-1] = label
    if state_vect:
        return Y, Y2
    else:
        return Y

def get_labels(f):
	pass

def import_data(n_files=None):
    # define root, filepath
    home = os.environ["HOME"]
    data_root = home+"/data/karaOne/"
    # get files
    data_dirs = sorted([data_root+f+'/' for f in os.listdir(data_root)
            if f[:2] in ('MM','P0')])[:n_files]
    #X files
    mapper = lambda root: [root+f for f in os.listdir(root) if f[-4:]=='.cnt'][0];
    cnt_files = sorted(map(mapper, data_dirs))
    #Y index files
    mapper = lambda root: [root+f for f in os.listdir(root) if f=='epoch_inds.mat'][0]
    ind_files = sorted(map(mapper, data_dirs))
    #Y label files 
    mapper_pre = lambda root: [root+f+'/' for f in os.listdir(root) if f=='kinect_data'][0]
    mapper = lambda root: [mapper_pre(root)+f for f in os.listdir(mapper_pre(root)) 
    				if (f[-4:]=='.txt' and len(f) in (7,8))][0]
    lab_files = map(mapper, data_dirs)
    # process X
    channel_range = np.arange(0,64)
    channel_dict = {name:i for i,name in enumerate(
            mne.io.read_raw_cnt(cnt_files[0], None, date_format="dd/mm/yy").ch_names[:len(channel_range)])}
    def m0(f):
        obj = mne.io.read_raw_cnt(f, None, date_format="dd/mm/yy")
        df = obj.to_data_frame()
        return df.iloc[:,channel_range]
    X_series = map(m0, cnt_files)
    # process Y
    epoch_dict = {'clearing_inds': 1, 'thinking_inds': 2, 'speaking_inds':3}
    # get epochwave
    mapper = lambda f,x : get_epochwave(f, x.shape[0])
    Y_epochwave = map(mapper, ind_files, X_series)
    # get labels
    Y_labels = get_labels(lab_files)
    return X_series, Y_epochwave

















