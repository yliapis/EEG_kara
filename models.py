from preprocessing import *


class MACE_filter:
    
    def __init__(self, fs=200):
        self.fs = fs
        pass
    
    def fit(self, X_win, Y):
        # H = D.I X (X+ D X).I u
        #
        # Lexicographic reordering and vetorizing input
        # just
        Xf = map(lambda x: x.reshape(-1), map(dft2, X_win))
        X = np.array(Xf).T
        mu_psd = sum(map(lambda x: (np.abs(x)**2), Xf))/len(Xf)
        D = mu_psd.reshape((-1,1))
        D_I = 1/D
        #
        u = np.array(Y).reshape(-1,1)
        #
        a = D_I * X
        b = np.linalg.inv(np.conj(X.T).dot(D*X))
        c = u
        print a.shape, b.shape, c.shape
        H = ( a ).dot( b ).dot( c )
        #
        self.H = H
        #
        return self

    def predict(self, X_win, method="frequency"):
    	if method == "frequency":
    		H = self.H.squeeze()
    		mapper = lambda win: np.abs(fftpack.fft2(win).reshape(-1).dot(H))
			return map(mapper, X_win)
    	else:
    		raise NotImplementedError()
        