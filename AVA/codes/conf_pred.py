# split-conformal functions

import torch
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles
from bbox import predict


        
        
class CQR:
  def __init__(self, bbox=None):
    if bbox is not None:
      self.bbox = bbox


  def calibrate(self, calib_loader, alpha, no_calib=False, return_scores=False):

    prediction , true_Y = predict(self.bbox, calib_loader, return_Y=True)

    prediction = torch.stack(prediction).cpu().numpy().squeeze(1)
    y_lower = prediction[:,0]
    y_upper = prediction[:,-1]
    
    error_high = true_Y.squeeze(1) - y_upper 
    error_low = y_lower - true_Y.squeeze(1)
    
    err_high = np.reshape(error_high, (y_upper.shape[0],1))
    err_low = np.reshape(error_low, (y_lower.shape[0],1))

    E = np.maximum(err_high, err_low)
    E = E + 1e-10*np.random.randn(E.size)

    n2 = len(calib_loader.dataset)

    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    Q = mquantiles(E, prob=level_adjusted)[0]
      
    self.Q = Q
    if no_calib:
      self.Q = 0
      
    if return_scores:
      return E, Q


  def predict(self, data_loader):

    prediction = predict(self.bbox, data_loader)
    prediction = torch.stack(prediction).cpu().numpy().squeeze(1)
    y_lower = prediction[:,0]
    y_upper = prediction[:,-1]

    Q = self.Q
    C = np.asarray([y_lower-Q, y_upper+Q]).T
    print(C.shape)
    return C    




class ResCalib:
  def __init__(self, bbox=None):
    if bbox is not None:
      self.bbox = bbox


  def calibrate(self, calib_loader, alpha, no_calib=False, return_scores=False):

    pred_mean , true_Y = predict(self.bbox, calib_loader, return_Y=True)
    
    n2 = len(calib_loader.dataset)
    residuals = np.abs(true_Y.squeeze(1) - torch.tensor(pred_mean).cpu().numpy())
    print(residuals.size)
    residuals = residuals + 1e-10*np.random.randn(residuals.size)
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    Q = mquantiles(residuals, prob=level_adjusted)[0]
      
    self.Q = Q
    if no_calib:
      self.Q = 0
      
    if return_scores:
      return residuals, Q


  def predict(self, data_loader):

    mean_pred = predict(self.bbox, data_loader)

    Q = self.Q
    C = np.asarray([torch.tensor(mean_pred).cpu().numpy()-Q, torch.tensor(mean_pred).cpu().numpy()+Q]).T
    return C
