from __future__ import division

import numpy as np

try:
    import cv2
except:
    print 'no cv2... continuing...'

from scipy import io
from scipy import signal
from scipy import fftpack

import pywt


def bp_filter(data, low, high, fs=1, order=5):
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = signal.butter(order, [low, high], btype='band')
            return b, a
        #
        b, a = butter_bandpass(low, high, fs, order=order)
        # signal is t x ch
        y = signal.lfilter(b, a, data, axis=0)
        return y


def rec_map(function, data, depth=2, num_threads=1):

    def recursive_mapper(function, data, depth):
        if depth == 1:
            return map(function, data)
        else:
            mapper = lambda d: recursive_mapper(function, d, depth-1)
            return map(mapper, data)
    if num_threads != 1:
        raise NotImplementedError()
    else:
        return recursive_mapper(function, data, depth)


def window_series(X, Y=None, width=100, stride=1, fs=1000, ordering="ch x t", win_label=False, num_wins=None):
    # scaling, assuming fs in kHz
    nwidth = int(width*fs/1000)
    nstride = max(1, int(stride*fs/1000))
    nlen = len(X)
    # generate indices
    starts = np.arange(0, nlen-nwidth, nstride, dtype=int)
    ends = np.arange(nwidth, nlen, nstride, dtype=int)
    if ordering == "ch x t":
        X_ = [np.copy(X[start:end].T) for start,end in zip(starts, ends)]
    elif ordering == "t x ch":
        X_ = [np.copy(X[start:end]) for start,end in zip(starts, ends)]
    else:
        raise NotImplementedError() 
    #
    if Y is not None:
        Y_ = np.array([Y[end] for end in ends])
        return X_[:num_wins], Y_[:num_wins]
    else:
        return X_[:num_wins]


def flatten(x):
    assert str(type(x)) == "<type 'numpy.ndarray'>", "not a numpy array..."
    return x.flatten()


def demean(X, mode='global'):
    # assuming a time window input
    if mode=='global':
        return X-X.mean()
    elif mode=='channel':
        return X-X.mean(1)[:,None]


def normalize(X, axis=0):
    assert np.ndim(X) == 2
    if axis is None:
        mean = np.mean(X)
        std = np.std(X)
    else:
        mean = np.mean(X, axis=axis)
        std = np.std(X, axis=axis)
    mapper = lambda f: (f-mean)/std
    return map(mapper, X)

# def window_dft_cv2old(win, frange=(0,40), fs=1000, demean_mode=None):
#   start, end = frange #* 1000/fs
#   if demean_mode is not None:
#       win = demean(win, mode=demean_mode)
#   dft = cv2.dft(win, flags=cv2.DFT_ROWS)[:,start:end]
#   return dft


def psd(win, ftype='DPSS'):
    if ftype is 'welch':
        f,P = signal.welch(win, fs=200, nperseg=100)
    elif ftype is 'periodogram':
        f,P = signal.periodogram(win, fs=200)
    else:
        P = []
        low, high = 0, 40
        for x in win:
            psd, f = mtspec.mtspec(x, 1/200, 2)
            F.append(f)
            P.append(psd)
        idxs = (low<=f)*(f<=high)
        return np.array(P)[:,idxs]
        
    low, high = 0, 40
    idxs = (low<=f)*(f<=high)
    return P[:,idxs]


def normalizer(feats):
    mu, std = np.mean(feats, 0), np.std(feats, 0)
    mapper = lambda f: (f-mu)/std
    return map(mapper, feats)   


def dft(x):
    return fftpack.fft(x)


def dft2(x):
    return fftpack.fft2(x)


def dwt(x, wavelet='coif1'):
    return np.concatenate(pywt.wavedec(x, wavelet))


def window_dft(win, frange=(0,40), fs=None):
    if fs is None: #should change... blah
        lims = (0, None)
    else:
        # assuming ch x t
        sz_win = win.shape[-1]
    scale = sz_win/(fs)
    lims = map(int, (frange[0]*scale, frange[1]*scale))
    winf = fftpack.fft(win, axis=-1)[:,lims[0]:lims[1]]
    return winf

def window_psd(win, frange=(0,40), fs=1000, mode='scipy', demean_mode=None, debug=False, normalize=False):
    if debug:
        print win.shape
    if demean_mode is not None:
        win = demean(win, mode=demean_mode)
    if mode == 'manual':
        N = win.shape[-1]
        
        dft = window_dft(win, frange, fs)   
        return dft ** 2 / N
    if mode == 'scipy':
        f, psd = signal.periodogram(win, fs=fs, axis=-1)
        if normalize:
            psd = psd / psd.sum(-1)
        if frange is None:
            return psd
        else:
            fmin, fmax = frange
            return psd[:,(f>=fmin) * (f<=fmax)]
    else:
        raise NotImplementedError()


def get_features(data_dir, ftype):
    f = None
    if ftype == "ICA all":
        f = "all_features_ICA.mat"
    elif ftype == "noICA all":
        f = "all_features_noICA.mat"
    elif ftype == "simple":
        f = "all_features_simple.mat"
    else:
        assert False
    return io.loadmat(data_dir+f)


def feature_vect(wins, ftype=1, fs=200, frange=(0,40),
    trim=False, norm=False, flatten=True):
    if ftype == 1: # psd
        window_size = wins[0].shape[-1]
        hann = signal.hann(window_size).reshape(1,-1)
        psd_mapper = lambda win: window_psd(win*hann, 
                                fs=fs, frange=frange)
        feats = map(psd_mapper, wins)
    elif ftype == 2:
        feats = wins
    elif ftype == 3: # dwt wavelets
        wtype = 'coif2'
        get_wavelets = lambda chann : pywt.wavedec(chann, wtype)
        flattener = lambda chann : [i for j in get_wavelets(chann) for i in j]
        wt_mapper = lambda win : np.array(map(flattener, win)).reshape(-1) # assuming ch x series
        feats = map(wt_mapper, wins)
    elif ftype == 4: # dft, real and comlex components
        # dft
        window_size = wins[0].shape[-1]
        hann = signal.hann(window_size).reshape(1,-1)
        dft_mapper = lambda win: window_dft(win*hann, 
                                fs=fs, frange=(0,30)).reshape(-1)
        dfts = map(dft_mapper, wins)
        # separate complex, real
        mapper = lambda v: np.concatenate([np.real(v), np.imag(v)])
        feats = map(mapper, dfts)
    elif ftype == 5: #dft, phase and magnitude
        # dft
        window_size = wins[0].shape[-1]
        hann = signal.hann(window_size).reshape(1,-1)
        dft_mapper = lambda win: window_dft(win*hann, 
                                fs=fs, frange=(0,30)).reshape(-1)
        dfts = map(dft_mapper, wins)
        # phase and magnitude
        mapper = lambda v: np.concatenate([np.abs(v), np.angle(v)])
        feats = map(mapper, dfts)
    elif ftype == 6: # 10 part spectral entropy
        window_size = wins[0].shape[-1]
        def mapper(win):
            # spectrogram
            f, t, sp = signal.spectrogram(win, fs=fs, nperseg=100, noverlap=40)
            sg = sp[:,f<40,:]
            # normalize
            nsg = sg/sg.mean(1)[:,None,:]
            # entropy
            pse = -np.sum( nsg * np.log(nsg) ,axis=1)
            return pse
        feats = map(mapper, wins)
    elif ftype == 7:
        mapper = lambda win: win.mean(1)
        feats = map(mapper, wins)
    elif ftype == 8: #minmax
        mapper = lambda win: win.max(-1)-win.min(-1)
        feats = map(mapper, wins)
    elif ftype == 9: # psd normed
        window_size = wins[0].shape[-1]
        hann = signal.hann(window_size).reshape(1,-1)
        psd_mapper = lambda win: window_psd(win*hann, 
                                fs=fs, frange=frange)
        feats = map(psd_mapper, wins)
        # norm psd
        mapper = lambda P: P/np.sum(P, 1)[:,None]
        feats = map(mapper, feats)
    elif ftype == 10: # welch psd
        # z norm
        #znorm = lambda w: (w-np.mean(w,1)[:,None]) / np.std(w,1)[:,None]
        #wins = map(znorm, wins)
        def psd(win):
            f, P = signal.welch(win, fs=fs, nperseg=100, noverlap=50)
            return P[:,(f>=frange[0])*(f<=frange[1])]
        feats = map(psd, wins)        
    elif ftype == 11:  # time series...?
        pass
    elif ftype == 12:  # log psd
        return feature_vect(wins, ftype=10)
    elif ftype == 13:  # log psd
        v = feature_vect(wins, ftype=10)
        return np.concatenate([v, np.log(v)], axis=-1)
    # flatten
    if flatten:
        mapper = lambda v: v.reshape(-1)
        feats = map(mapper, feats)
    # norm
    if norm:
        mean = np.mean(feats, 0)
        std = np.std(feats, 0)
        mapper = lambda f: (f-mean)/std
        feats = map(mapper, feats)
    return feats

#### pipelines
def pipeline_standard(X_raw, depth=3, features='psd'):
    # window dft
    if features == 'dft':
        mapper = lambda x : window_dft(x, demean_mode='global')
    elif features == 'psd':
        mapper = lambda x : window_psd(x, demean_mode='global')
    X = rec_map(mapper, X_raw, depth)
    # flatten
    X = rec_map(flatten, X, depth)
    #
    return X


from itertools import chain,combinations
def powerset_gen(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    




