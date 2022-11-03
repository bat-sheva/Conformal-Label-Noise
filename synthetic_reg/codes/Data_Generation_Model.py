import numpy as np


def f(x):
    ''' Construct data (1D example)
    '''
    ax = np.zeros(x.shape[0],)
    for i in range(x.shape[0]):
        ax[i] = np.random.poisson(np.sin(np.mean(x[i,:]))**2+0.1) + 0.03*np.mean(x[i,:])*np.random.randn(1)
        ax[i] += 25*(np.random.uniform(0,1,1)<0.01)*np.random.randn(1)
    return ax.astype(np.float32)
