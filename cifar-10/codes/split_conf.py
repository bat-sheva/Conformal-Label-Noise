# split-conformal functions

import torch
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles
from black_boxes_CNN import predict, predict_proba


# The HPS non-conformity score
def class_probability_score(probabilities, labels, all_combinations=False):

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # calculate scores of each point with all labels
    if all_combinations:
        scores = 1 - probabilities[:, labels]

    # calculate scores of each point with only one label
    else:
        scores = 1 - probabilities[np.arange(num_of_points), labels]

    # return scores
    return scores
    
    
#class ProbAccum:
#    def __init__(self, prob):
#        prob = np.clip(prob,1e-6,1.0-2e-6)
#        prob += np.random.uniform(low=0.0, high=1e-6, size=prob.shape)
#        prob /= np.sum(prob,1)[:,None]
#        self.prob = prob
#        self.n, self.K = prob.shape
#        self.order = np.argsort(-prob, axis=1)
#        self.ranks = np.empty_like(self.order)
#        for i in range(self.n):
#            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
#        self.prob_sort = -np.sort(-prob, axis=1)
#        #self.epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
#        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)        
#        
#    def predict_sets(self, alpha, randomize=False, epsilon=None):
#        L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
#        idx_ra0 = np.where(L==0)[0]
#        if randomize:
#            epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
#        if epsilon is not None:
#            epsilon[idx_ra0] = 1
#            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
#            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
#            p_remove = np.clip(p_remove, 0, 1)
#            remove = np.where(epsilon < p_remove)[0]
#            L[remove] -= 1
#            # Make sure no prediction sets are empty
#            L[L<0] = 0
#        # Return prediction set
#        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
#        return(S)
#
#    def calibrate_scores(self, Y, epsilon=None, scores='RAPS'):
#        Y = np.atleast_1d(Y)
#        n2 = len(Y)
#        if scores=='RAPS':
#          ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
#          prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
#          prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
#          alpha_max = 1.0 - prob_cum
#          
#          if epsilon is not None:
#            # This will give exact marginal coverage
#            idx_ra0 = np.where(ranks==0)[0]
#            epsilon[idx_ra0] = 1
#            
#            alpha_max += np.multiply(prob, epsilon)
#          else:
#              alpha_max += prob
#  
#          alpha_max = np.minimum(alpha_max, 1)
#          return alpha_max
#  
#        if scores=='HPS':
#          return class_probability_score(self.prob, Y, all_combinations=False)


class ProbAccum:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        #self.epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)        
        
    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        if epsilon is not None:
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1

        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max
        
        

class SplitConformal:
  def __init__(self, bbox=None):
    if bbox is not None:
      self.bbox = bbox


  def calibrate(self, calib_loader, alpha, bbox=None, return_scores=False, no_calib=False, scores='RAPS', noise_tr=False):
    if bbox is not None:
        self.bbox = bbox
    self.noise_tr = noise_tr

    # Form prediction sets on calibration data

    if noise_tr:
      p_hat_calib, Y = self.bbox.predict_proba(calib_loader, return_y_true = True)
    else:
      p_hat_calib, Y = predict_proba(self.bbox, calib_loader, return_y_true = True)
    
    grey_box = ProbAccum(p_hat_calib)

    n2 = len(calib_loader.dataset)
          
    epsilon = np.random.uniform(low=0.0, high=1.0, size=n2)
    scores = grey_box.calibrate_scores(Y, epsilon=epsilon)
    level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
    tau = mquantiles(1.0-scores, prob=level_adjusted)[0]

    if return_scores:
      self.scores = scores
      
    # Store calibrate level
    self.alpha_calibrated = 1.0 - tau
    if no_calib:
      self.alpha_calibrated = alpha
    print("Calibrated alpha nominal {:.3f}: {:.3f}".format(alpha, self.alpha_calibrated))


  def predict(self, data_loader, alpha=None, epsilon=None):
    n = len(data_loader.dataset)
    if epsilon is None:
      epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
    if self.noise_tr:
      p_hat = self.bbox.predict_proba(data_loader)
    else:
      p_hat = predict_proba(self.bbox, data_loader)
    grey_box = ProbAccum(p_hat)
    if alpha is None:
      alpha = self.alpha_calibrated
    S_hat = grey_box.predict_sets(alpha, epsilon=epsilon)
    return S_hat


#class SplitConformal:
#    def __init__(self, X, Y, black_box, alpha, random_state=2020, allow_empty=True, verbose=False):
#        self.allow_empty = allow_empty
#
#        # Split data into training/calibration sets
#        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)
#        n2 = X_calib.shape[0]
#
#        self.black_box = black_box
#
#        # Fit model
#        self.black_box.fit(X_train, Y_train)
#
#        # Form prediction sets on calibration data
#        p_hat_calib = self.black_box.predict_proba(X_calib)
#        grey_box = ProbAccum(p_hat_calib)
#
#        rng = np.random.default_rng(random_state)
#        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
#        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
#        scores = alpha - alpha_max
#        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
#        alpha_correction = mquantiles(scores, prob=level_adjusted)
#
#        # Store calibrate level
#        self.alpha_calibrated = alpha - alpha_correction
#
#    def predict(self, X, random_state=2020):
#        n = X.shape[0]
#        rng = np.random.default_rng(random_state)
#        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
#        p_hat = self.black_box.predict_proba(X)
#        grey_box = ProbAccum(p_hat)
#        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
#        return S_hat
        
        



def evaluate_predictions(S, y, noisy=False, clean=False):
  # Marginal coverage
  marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    
  # Size and size conditional on coverage
  size = np.mean([len(S[i]) for i in range(len(y))])     
  size_median = np.median([len(S[i]) for i in range(len(y))])
  idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
  size_cover = np.mean([len(S[i]) for i in idx_cover])
  # Combine results
  if noisy:
    out = pd.DataFrame({'Coverage_noisy': [marg_coverage],
                        'Size_noisy': [size], 'Size_noisy (median)': [size_median], 
                        'Size_noisy conditional on cover': [size_cover]})
                        
  elif clean:
    out = pd.DataFrame({'Coverage_clean': [marg_coverage],
                        'Size_clean': [size], 'Size_clean (median)': [size_median], 
                        'Size_clean conditional on cover': [size_cover]})
  
  else:
    out = pd.DataFrame({'Coverage': [marg_coverage],
                        'Size': [size], 'Size (median)': [size_median], 
                        'Size conditional on cover': [size_cover]})
  return out