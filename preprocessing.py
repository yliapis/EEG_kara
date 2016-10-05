from __future__ import division

import numpy as np

try:
	import cv2
except:
	print 'no cv2... continuing...'

from scipy import io
from scipy import signal
from scipy import fftpack


def rec_map(function, data, depth=4, num_threads=1):
    
    heirarchy = ["subject",
                 "triples",
                 "pairs",
                 "data"]
    
    def recursive_mapper(function, data, depth):
        if depth == 1:
            return map(function, data)
        else:
            mapper = lambda d: recursive_mapper(function, d, depth-1)
            return map(mapper, data)
    
    if num_threads>1:
        raise NotImplementedError()
    else:
        return recursive_mapper(function, data, depth)


def window_series(X, Y=None, width=100, nstride=1, fs=1000, ordering="ch x t", win_label=False):
	# scaling, assuming fs in kHz
	nwidth = int(width*fs/1000)
	nlen = len(X)
	# generate indices
	starts = np.arange(0, nlen-nwidth, nstride, dtype=int)
	ends = np.arange(nwidth, nlen, nstride, dtype=int)
	if ordering == "ch x t":
		X_ = [X[start:end].T for start,end in zip(starts, ends)]
	elif ordering == "t x ch":
		X_ = [X[start:end] for start,end in zip(starts, ends)]
	else:
		raise NotImplementedError()	
	#
	if Y is not None:
		Y_ = np.array([Y[end] for end in ends])
		return X_, Y_
	else:
		return X_


def flatten(x):
    assert str(type(x)) == "<type 'numpy.ndarray'>", "not a numpy array..."
    return x.flatten()


def demean(X, mode='global'):
	# assuming a time window input
	if mode=='global':
		return X-X.mean()
	elif mode=='channel':
		return X-X.mean(1)[:,None]


# def window_dft_cv2old(win, frange=(0,40), fs=1000, demean_mode=None):
# 	start, end = frange #* 1000/fs
# 	if demean_mode is not None:
# 		win = demean(win, mode=demean_mode)
# 	dft = cv2.dft(win, flags=cv2.DFT_ROWS)[:,start:end]
# 	return dft


def dft(x):
	return fftpack.fft(x)


def dft2(x):
	return fftpack.fft2(x)


def window_dft(win):
	return fftpack.fft(win, axis=-1)


def window_psd(win, frange=(0,40), fs=1000, mode='scipy', demean_mode=None):
	if mode == 'manual':
		N = win.shape[-1]
		if demean_mode is not None:
			win = demean(win, mode=demean_mode)
		dft = window_dft(win, frange, fs)	
		return dft ** 2 / N
	if mode == 'scipy':
		f, psd = signal.periodogram(win, fs=fs, axis=-1)
		if frange is None:
			return psd
		else:
			fmin, fmax = frange
			return psd[(f>=fmin) * (f<=fmax)]
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
