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


def get_labels(f, offset=0, use_dict=True):
    #
    data_list = open(f, 'r').read().replace('\r','').split('\n')
    filt = lambda x: x != ''
    data_list = filter(filt, data_list)
    #
    if use_dict: #good idea to use the same dictionary across all subjects
        Y_dict = {
            '/m/ mmm' : 0,
            '/n/ nnn' : 1,
            '/tiy/ tee' : 2,
            '/piy/ pea' : 3,
            '/diy/ dee' : 4,
            '/iy/ ee' : 5,
            '/uw/ ooh' : 6,
            'knew' : 7,
            'gnaw' : 8,
            'pat' : 9,
            'pot' : 10
        }
        for key in Y_dict.keys():
            Y_dict[key] += offset
    else:
        Y_dict = {}
        label_count = 0+offset
        for label in data_list:
            if label not in Y_dict.keys():
                Y_dict[label] = label_count
                label_count += 1
    #
    Y = np.array([Y_dict[i] for i in data_list])
    return Y, Y_dict


def get_data_dirs(n_files=None):
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
    lab_files = sorted(map(mapper, data_dirs))
    #
    return cnt_files, ind_files, lab_files


def import_data(n_files=None):
    # get file locations
    cnt_files, ind_files, lab_files = get_data_dirs(n_files)
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

















