from __future__ import division

import numpy as np
import os
import mne

from scipy import io, signal
from preprocessing import window_series, rec_map, bp_filter
from keras.utils import np_utils
from multiprocessing import Pool



def get_statewave(f, size=None, state_vect=False):
    label_dict = {'clearing_inds': 1, 'thinking_inds': 3, 'speaking_inds': 4}
    ind_dict = io.loadmat(f)
    # format indices
    for key in label_dict.keys():
        ind_dict[key] = np.array(ind_dict[key].tolist()).squeeze()
    # split speaking inds
    if True:
        label_dict['hearing_inds'] = 2
        ind_dict['hearing_inds'] = ind_dict['speaking_inds'][::2]
        ind_dict['speaking_inds'] = ind_dict['speaking_inds'][1::2]
    # split items
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
    num_ewave = int(epochwave[-1]-epochwave[0]+1)
    if int(num_ewave) == len(labels):
        print "inconsistent number of epochs and labels:\
                        \n\t expected:{} have:{}".format(num_ewave, len(labels))
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
    if use_dict: # good idea to use the same dictionary across all subjects
        Y_dict = {
            '/m/ mmm': 0,
            '/n/ nnn': 1,
            '/tiy/ tee': 2,
            '/piy/ pea': 3,
            '/diy/ dee': 4,
            '/iy/ ee': 5,
            '/uw/ ooh': 6,
            'knew': 7,
            'gnaw': 8,
            'pat': 9,
            'pot': 10
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


def class_filter(X, Y, filt, mode='epoch'):
    if filt=='letter':
        filt = np.arange(0, 7)
    elif filt=='word':
        filt = np.arange(7, 11)
    elif filt is None:
        return X, Y
    # 
    if mode == 'epoch':
        X_filt, Y_filt = [], []
        for xs, ys in zip(X,Y):
            x0, y0 = [], []
            for xe, ye in zip(xs,ys):
                if ye[0] in filt:
                    x0.append(xe)
                    y0.append(ye)
            X_filt.append(x0)
            Y_filt.append(y0)
        #
        return X_filt, Y_filt
    else:
        raise NotImplementedError()


def get_data_dirs(n_files=None, simple=False):
    # define root, filepath
    home = os.environ["HOME"]
    data_root = home+"/data/karaOne/"
    # get files
    data_dirs = sorted([data_root+f+'/' for f in os.listdir(data_root)
            if f[:2] in ('MM', 'P0')])[:n_files]
    if simple:
        return data_dirs
    # X files
    mapper = lambda root: [root+f for f in os.listdir(root) if f[-4:] == '.cnt'][0]
    cnt_files = sorted(map(mapper, data_dirs))
    # Y index files
    mapper = lambda root: [root+f for f in os.listdir(root) if f == 'epoch_inds.mat'][0]
    ind_files = sorted(map(mapper, data_dirs))
    # Y label files
    mapper_pre = lambda root: [root+f+'/' for f in os.listdir(root) if f=='kinect_data'][0]
    mapper = lambda root: [mapper_pre(root)+f for f in os.listdir(mapper_pre(root))
    						if (f[-4:]=='.txt' and len(f) in (7,8))][0]
    lab_files = sorted(map(mapper, data_dirs))
    #
    return cnt_files, ind_files, lab_files


def narrow(y, label, offset=0, duration=2, fs=200):
    assert type(y) == type(np.arange(9))
    yl = (y==label).astype(int)
    dy = np.diff(yl)
    starts, ends = np.where(dy>0)[0], np.where(dy<0)[0]
    starts = starts + int(offset*fs)
    ends = np.copy(starts) + int(duration*fs)
    y_new = np.zeros(y.shape)
    for s,e in zip(starts, ends):
        y_new[s:e] = label
    return y - yl*label + y_new
    

def import_data(n_files=None, preprocessor=None, downsample=None, ch_dict=False, 
    ICA=False, bp_range=(.5, 40), channel_range="valid", filt=None):
    # we know fs
    fs = 1000
    fs /= downsample
    # get file locations
    cnt_files, ind_files, lab_files = get_data_dirs(n_files)
    # process X
    channel_dict = {name:i for i,name in enumerate(
            mne.io.read_raw_cnt(cnt_files[0], None, date_format="dd/mm/yy").ch_names)}
    def m0(f, channel_range=channel_range):
        raw = mne.io.read_raw_cnt(f, None, date_format="dd/mm/yy")

        if not ICA:
            dfx = raw.to_data_frame()
            dfx = dfx.values
            if channel_range == "valid":
                channel_range = range(0,64)
                channel_range.remove(42) #M2
                channel_range.remove(32) #M1
                dfx = dfx[:,channel_range]               
        else:
            # perform ICA
            raise NotImplementedError()
        return dfx
    # define bandpass filter
    
    #
    X_series = map(m0, cnt_files)
    if filt=='CAR': # commont average reference
        mapper = lambda x: x - x.mean(1)[:,None]
        X_series = map(mapper, X_series)
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
    if downsample is not None:
        assert downsample%1 == 0, "downsample should be an int"
        mapper = lambda y: y[downsample-1::downsample]
        Y_statewave = map(mapper, Y_statewave)
        Y_labelwave = map(mapper, Y_labelwave)
        mapper = lambda x: signal.decimate(x, downsample, axis=0, zero_phase=True)
        X_series = map(mapper, X_series)
    #
    if bp_range is not None:
        low, high = bp_range
        mapper = lambda series : bp_filter(series, low, high, fs=fs)
        X_series = map(mapper, X_series)
    if ch_dict:
        return X_series, Y_statewave, Y_labelwave, channel_dict
    else:
        return X_series, Y_statewave, Y_labelwave


def import_data_chunks(n_files=None, state="thinking_inds", preprocessor=None, downsample=None, bp_range=(1,40)):
    # use original file initially
    X_series, Y_statewave, Y_labelwave = import_data(n_files, preprocessor, downsample=downsample, bp_range=bp_range)
    ind_files = get_data_dirs(n_files)[1]
    if state == "speaking_inds":
        raise NotImplementedError("not implemented for speaking_inds...")
    # get index limits
    def get_idxs(f, state=state):
        ind_dict = io.loadmat(f)
        return np.array(ind_dict[state].tolist()).squeeze()
    idxs = map(get_idxs, ind_files)
    if downsample is not None:
        idxs = map(lambda x: (x/downsample).astype(int), idxs)
    def segment(series, idxs):
        return [series[idx0-1:idxf+1] for idx0,idxf in idxs]
    X = map(segment, X_series, idxs)
    Y = map(segment, Y_labelwave, idxs)
    return X, Y


def train_test_split(X, Y, valid=False):
    '''
        X contains instances of epoch windows
        Y is assumed to contain equal sized windows of epoch labels
    '''
    # first organize data
    epoch_dict = {}
    dim(X), dim(Y)
    for x,y in zip(X,Y):
        label = y[0]
        if label not in epoch_dict:
            epoch_dict[label] = [x]
        else:
            epoch_dict[label].append(x)
    # assuming keep two epochs each for testing, 1 for validation : n:1:2 split
    # idxs
    train0, trainf = 0, -3
    if valid:
        valid0, validf = -3, -2
        test0, testf = -2, None
    else:
        valid0, validf = None, None
        test0, testf = -3, None
    #
    train, valid, test = [], [], []
    mapper = lambda x, y: y*np.ones(len(x))
    for y,x in epoch_dict.items():
        data = zip(x, map(mapper, x, [y for _ in range(len(x))]))
        #splits
        #train
        train.extend(data[train0:trainf])
        #valid
        valid.extend(data[valid0:validf])
        #test
        test.extend(data[test0:testf])        
    #
    mapper = lambda T: zip(*T)
    train, valid, test = map(mapper, (train, valid, test))
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = train, valid, test
    return (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test)


def state_window(X, Y, rest=1, label=3, fs=200, window_size=2000, window_stride=200, num_wins=1):
    # locate labels
    Y_1 = (Y == label).astype(int)
    Y_0 = (Y == rest).astype(int)
    diff0, diff1 = np.diff(Y_0), np.diff(Y_1)
    # compute window limits
    start0, end0 = map(lambda x: np.where(x)[0], [diff0>0, diff0<0])
    start1, end1 = map(lambda x: np.where(x)[0], [diff1>0, diff1<0])

    mapper = lambda lims: X[lims[0]:lims[1]]
    X0, X1 = rec_map(mapper, [zip(start0, end0), zip(start1, end1)], 2)
    # window series
    mapper = lambda win: window_series(win, width=window_size, 
                                       stride=window_stride, fs=fs, num_wins=num_wins)
    X0, X1 = rec_map(mapper, [X0, X1], 2)
    # flatten windows
    mapper = lambda x: [i for j in x for i in j]
    X0, X1 = map(mapper, [X0, X1])
    # split data
    X0_train, X0_test = split(X0)
    X1_train, X1_test = split(X1)
    X_train, X_test = map(np.concatenate, [[X0_train, X1_train], [X0_test, X1_test]])
    # create label vectors
    Y0_train, Y0_test = np.zeros(len(X0_train)), np.zeros(len(X0_test))
    Y1_train, Y1_test = np.ones(len(X1_train)), np.ones(len(X1_test))
    Y_train, Y_test = map(np.concatenate, [[Y0_train, Y1_train], [Y0_test, Y1_test]])
    #
    return (X_train, Y_train), (X_test, Y_test)


#### import everything processed ...
def standard_import(n_files=None, width=400, stride=50, filt=None, downsample=None):
    fs = 1000
    if downsample is not None:
        fs = fs / downsample
    X, Y = import_data_chunks(n_files, downsample=downsample)
    # filter
    X, Y = class_filter(X, Y, filt=filt)
    # subj x (train valXid test) x stuff
    mapper = lambda X,Y: train_test_split(X, Y)
    X, Y = zip(*map(train_test_split, X, Y))
    ### window ###
    XX, YY = [], []
    for sx, sy in zip(X, Y):
        lx0, ly0 = [], []
        for x,y in zip(sx,sy):
            lx1, ly1 = [], []
            for xe, ye in zip(x,y):
                x_, y_ = window_series(xe, ye, width=width, fs=fs, stride=stride)
                lx1.extend(x_), ly1.extend(y_)
            lx0.append(lx1), ly0.append(ly1)
        XX.append(lx0), YY.append(ly0)
    ### end ###
    return XX, YY


def dim_collapse(data, level=1):
    if level == 0:
        return data
    else:
        new = []
        for d in data:
            new.extend(d)
        return dim_collapse(new, level=level-1)

# just to analyze dimensionality
def dim(x):
    try:
        print len(x),
        dim(x[0])
    except:
        print ""
        return


def to_cat(y):
    return np_utils.to_categorical(y)
def from_cat(y):
    return np_utils.categorical_probas_to_classes(y)


def balance(X, Y, mode='binary'):
    if mode == 'binary':
        #to np array
        Y = np.array(Y).astype(int)
        #find values
        Y_true, Y_null = (Y==0), (Y!=0)
        size = min(sum(Y_true), sum(Y_null))
        # find idxs and shuffle
        def mapper(Y_bool):
            Y_i = np.where(Y_bool)[0]
            sz = Y_i.shape[0]
            shuffler = lambda z: np.random.permutation(z)
            return Y_i[shuffler(sz)][shuffler(size)]
        Y_true_i, Y_null_i = map(mapper, [Y_true, Y_null])
        X = [X[i] for i in Y_true_i] + [X[i] for i in Y_null_i]
        Y = [Y[i] for i in Y_true_i] + [Y[i] for i in Y_null_i]
        shuffler = np.random.permutation(len(X))
        shuffle = lambda ls: [ls[i] for i in shuffler]
        return shuffle(X), shuffle(Y)
    else:
        raise NotImplementedError()


def split(x, split=.33):
    point = int(len(x)*(1-split))
    return np.copy(x[:point]), np.copy(x[point:])


def narrow(y, label, offset=0, duration=2, fs=200):
    assert type(y) == type(np.arange(9))
    yl = (y==label).astype(int)
    dy = np.diff(yl)
    starts, ends = np.where(dy>0)[0], np.where(dy<0)[0]
    starts = starts + int(offset*fs)
    ends = np.copy(starts) + int(duration*fs)
    y_new = np.zeros(y.shape)
    for s,e in zip(starts, ends):
        y_new[s:e] = label
    return y - yl*label + y_new
