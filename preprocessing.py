from __future__ import division

import numpy as np

from data_utils import *

def window(X, Y, width=100, nstride=1, fs=1000):
	# scaling, assuming fs in kHz
	nwidth = width*fs/1000
	nlen = len(X)
	# generate indices
	starts = np.arange(0, nlen-nwidth, nstride, dtype=int)
	ends = np.arange(nwidth, nlen, nstride, dtype=int)
	X_ = [X[start:end] for start,end in zip(starts, ends)]
	Y_ = [Y[end] for end in ends]
	return X_, Y_


def demean(X, mode='global'):
	# assuming a time window input
	if mode=='global':
		return X-X.mean()








#### pipelines

def pipeline_standard(X_raw, Y_raw):
	pass
