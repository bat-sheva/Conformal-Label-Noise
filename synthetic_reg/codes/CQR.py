# split-conformal functions

import torch
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles

      
      
      
class CQR_rf:
  def __init__(self, bbox=None):
    if bbox is not None:
      self.bbox = bbox


  def calibrate(self, x_calib, y_calib, alpha, no_calib=False, return_scores=False):

    predictions = self.bbox.predict(x_calib)
    y_lower = predictions[:,0]
    y_upper = predictions[:,-1]
    
    error_high = y_calib - y_upper 
    error_low = y_lower - y_calib
    
    err_high = np.reshape(error_high, (y_upper.shape[0],1))
    err_low = np.reshape(error_low, (y_lower.shape[0],1))

    E = np.maximum(err_high, err_low)

    n2 = len(y_calib)

    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    Q = mquantiles(E, prob=level_adjusted)[0]
      
    self.Q = Q
    if no_calib:
      self.Q = 0
    
    if return_scores:
        return E,Q


  def predict(self, X):

    predictions = self.bbox.predict(X)
    y_lower = predictions[:,0]
    y_upper = predictions[:,-1]

    Q = self.Q
    C = np.asarray([y_lower-Q, y_upper+Q]).T
    return C    