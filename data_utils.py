from __future__ import division

import numpy as np
import os
import mne
from scipy import io



def get_statewave(f, size=None, state_vect=False):
    label_dict = {'clearing_inds': 1, 'thinking_inds': 2, 'speaking_inds':3}
    ind_dict = io.loadmat(f)
    # format indices
    for key in label_dict.keys():
        ind_dict[key] = np.array(ind_dict[key].tolist()).squeeze()
    #
    if not size:
        size = np.max(np.concatenate([ind_dict[key].flatten() for key in label_dict.keys()]))
    Y = np.zeros(size)
    if state_vect:
        Y2 = np.zeros((size, 3))
    for key,val in label_dict.items():
        label = label_dict[key]
        for ind0,indf in ind_dict[key]:
            Y[ind0-1:indf] = label
            if state_vect:
                Y2[ind0-1:indf+1,label-1] = label
    if state_vect:
        return Y, Y2
    else:
        return Y



def get_epochwave(f, size=None):
    label_dict = {'clearing_inds': 1, #'thinking_inds': 2, 
                'speaking_inds':3}
    ind_dict = io.loadmat(f)
    # format indices
    for key in label_dict.keys():
        ind_dict[key] = np.array(ind_dict[key].tolist()).squeeze()
    # transform to epoch limits
    starts = ind_dict['clearing_inds'][:,0].squeeze()
    ends = ind_dict['speaking_inds'][1::2,1].squeeze()
    # make epochwave
    if not size:
        size = np.max(ends)
    epochwave = np.zeros(size)
    epochwave[starts[0]] = 1
    for i in ends[:-1]:
        epochwave[i] = 1
    #
    epochwave = np.cumsum(epochwave)
    return epochwave
    


def make_labelwave(epochwave, labels):
    assert epochwave[-1] == len(labels), "inconsistent number of epochs and labels"
    # for begining of trials, with no epoch
    labels = np.concatenate(([-1], labels))
    #
    labelwave = labels[epochwave.astype(int)]
    return labelwave



def get_labels(f, offset=0, use_dict=True, ret_dict=False):
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
    if ret_dict:
        return Y, Y_dict
    else:
        return Y



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



def import_data(n_files=None, preprocessor=None):
    # get file locations
    cnt_files, ind_files, lab_files = get_data_dirs(n_files)
    # process X
    channel_range = np.arange(0,64)
    channel_dict = {name:i for i,name in enumerate(
            mne.io.read_raw_cnt(cnt_files[0], None, date_format="dd/mm/yy").ch_names[:len(channel_range)])}
    def m0(f):
        obj = mne.io.read_raw_cnt(f, None, date_format="dd/mm/yy")
        df = obj.to_data_frame()
        return df.values[:,channel_range]
    X_series = map(m0, cnt_files)
    # get statewave
    mapper = lambda f,x : get_statewave(f, x.shape[0])
    Y_statewave = map(mapper, ind_files, X_series)
    # get labelwave
    mapper = lambda f_ind,f_lab, x: make_labelwave(get_epochwave(f_ind, size=x.shape[0]), 
                                                    get_labels(f_lab))
    Y_labelwave = map(mapper, ind_files, lab_files, X_series)
    # preprocessor for time series
    if preprocessor:
        X_series = preprocessor(X_series)
    #
    return X_series, Y_statewave, Y_labelwave



### this function is to get time series chunks
def import_data_windows(n_files=None, state="thinking_inds", preprocessor=None):
    # use original file initially
    X_series, Y_statewave, Y_labelwave = import_data(n_files, preprocessor)
    ind_files = get_data_dirs(n_files)[1]
    if state == "speaking_inds":
        raise NotImplementedError("not implemented for speaking_inds...")
    # get index limits
    def get_idxs(f, state=state):
        ind_dict = io.loadmat(f)
        return np.array(ind_dict[state].tolist()).squeeze()
    idxs = map(get_idxs, ind_files)
    def segment(series, idxs):
        return [series[idx0-1:idxf+1] for idx0,idxf in idxs]
    X = map(segment, X_series, idxs)
    Y = map(segment, Y_labelwave, idxs)
    return X, Y



###
def train_test_split(X, Y):



    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)









