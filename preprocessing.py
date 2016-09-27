from __future__ import division

import numpy as np

from data_utils import rec_map
from data_utils import *

import cv2


def window_series(X, Y=None, width=100, nstride=1, fs=1000, ordering="ch x t"):
	# scaling, assuming fs in kHz
	nwidth = width#int(width*fs/1000)
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
		Y_ = [Y[end] for end in ends]
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



def window_dft(win, frange=(0,40), fs=1000, demean_mode=None):
	start, end = frange #* 1000/fs
	if demean_mode is not None:
		win = demean(win, mode=demean_mode)
	dft = cv2.dft(win, flags=cv2.DFT_ROWS)[:,start:end]
	return dft
	


#### pipelines

def pipeline_standard(X_raw, depth=3):
	# window dft
	mapper = lambda x : window_dft(x, demean_mode='global')
	X = rec_map(mapper, X_raw, depth)
	# flatten
	X = rec_map(flatten, X, depth)
	#
	return X
