import os
import sys
import torch
import random
import pandas as pd
import numpy as np

#from cqr import helper
#from nonconformist.nc import RegressorNc
#from nonconformist.cp import IcpRegressor
#from nonconformist.nc import QuantileRegErrFunc

sys.path.append('./codes/')

from cqr import helper
from CQR import CQR_rf
from Data_Generation_Model import f
from reg_noise import add_Y_noise_gaussian, add_Y_noise_mean


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(seed, sigma):

    #seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    # number of training examples
    n_train = 8000
    # number of test examples (to evaluate average coverage and length)
    n_test = 5000
    # number of calib examples
    n_calib = 2000
    
    n_features = 100


    # training features
    x_train = np.random.uniform(0, 5.0, size=(n_train,n_features)).astype(np.float32)
    
    # test features
    x_test = np.random.uniform(0, 5.0, size=(n_test,n_features)).astype(np.float32)
    
    # calib features
    x_calib = np.random.uniform(0, 5.0, size=(n_calib,n_features)).astype(np.float32)
    
    # generate labels
    y_train = f(x_train)
    y_test = f(x_test)
    y_calib = f(x_calib)
    
    # add gaussian noise
    y_train_noisy_gaussian = add_Y_noise_gaussian(y_train, sigma=sigma, gaussian=True)
    y_calib_noisy_gaussian = add_Y_noise_gaussian(y_calib, sigma=sigma, gaussian=True)
    y_test_noisy_gaussian = add_Y_noise_gaussian(y_test, sigma=sigma, gaussian=True)
    
    y_train_noisy_t = add_Y_noise_gaussian(y_train, sigma=sigma)
    y_calib_noisy_t = add_Y_noise_gaussian(y_calib, sigma=sigma)
    y_test_noisy_t = add_Y_noise_gaussian(y_test, sigma=sigma)
    
    y_train_noisy_gumbel = add_Y_noise_gaussian(y_train, sigma=sigma, gumbel=True)
    y_calib_noisy_gumbel = add_Y_noise_gaussian(y_calib, sigma=sigma, gumbel=True)
    y_test_noisy_gumbel = add_Y_noise_gaussian(y_test, sigma=sigma, gumbel=True)
    
    y_train_noisy_positive = add_Y_noise_gaussian(y_train, positive=True, sigma=sigma)
    y_calib_noisy_positive = add_Y_noise_gaussian(y_calib, positive=True, sigma=sigma)
    y_test_noisy_positive = add_Y_noise_gaussian(y_test, positive=True, sigma=sigma)
    
    
    y_train_noisy_mean = add_Y_noise_mean(y_train)
    y_calib_noisy_mean = add_Y_noise_mean(y_calib)
    y_test_noisy_mean = add_Y_noise_mean(y_test)
    
    
    y_train_noisy_dispersive = add_Y_noise_mean(y_train, contractive=False)
    y_calib_noisy_dispersive = add_Y_noise_mean(y_calib, contractive=False)
    y_test_noisy_dispersive = add_Y_noise_mean(y_test, contractive=False)
    
    # reshape the features
    x_train = np.reshape(x_train,(n_train,n_features))
    x_test = np.reshape(x_test,(n_test,n_features))
    x_calib = np.reshape(x_calib,(n_calib,n_features))

    
    # desired miscoverage error
    alpha = 0.1

    # define quantile random forests (QRF) parameters
    params_qforest = dict()
    params_qforest["n_estimators"] = 200
    params_qforest["min_samples_leaf"] = 20
    params_qforest["max_features"] = n_features
    params_qforest["CV"] = True
    params_qforest["coverage_factor"] = 0.9
    params_qforest["test_ratio"] = 0.1
    params_qforest["random_state"] = seed
    params_qforest["range_vals"] = 10
    params_qforest["num_vals"] = 4
    
##################### train clean    
    
    
    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train)
    
    
    ############### t noise
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_t, alpha)
    predictions = cqr_conf_method.predict(x_test)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_t = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_t = np.mean(y_upper - y_lower)
    
    
    
    
    ############### gaussian noise
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_gaussian, alpha)
    predictions = cqr_conf_method.predict(x_test)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_gaussian = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_gaussian = np.mean(y_upper - y_lower)
    
    
    
    
    ############### gumbel noise
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_gumbel, alpha)
    predictions = cqr_conf_method.predict(x_test)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_gumbel = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_gumbel = np.mean(y_upper - y_lower)
    
    
    
    
    
   ############### positive noise
    
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_positive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_positive = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_positive = np.mean(y_upper - y_lower)
    
    
    
   ############### mean noise
    
   
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_mean, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_mean = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_mean = np.mean(y_upper - y_lower)
    
    
    
    ############## dispersive noise
    
   
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_dispersive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_dispersive = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_dispersive = np.mean(y_upper - y_lower)
    
    
    
    ### Wrong to right noise
    
    
 
    tmp_pred = quantile_estimator.predict(x_calib)
    y_lower_tmp = tmp_pred[:,0]
    y_upper_tmp = tmp_pred[:,1]
    
    right_idx = np.where((y_calib >= y_lower_tmp) & (y_calib <= y_upper_tmp))[0]
    wrong_idx = np.where((y_calib <= y_lower_tmp) | (y_calib >= y_upper_tmp))[0]
    y_calib_noisy_W2R = y_calib.copy()
    y_calib_noisy_W2R[wrong_idx[:int(0.07*len(y_calib))]] = y_calib_noisy_W2R[right_idx[:int(0.07*len(y_calib))]]
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_W2R, alpha)
    cqr_conf_predictions = cqr_conf_method.predict(x_test)
    

    y_lower = cqr_conf_predictions[:,0]
    y_upper = cqr_conf_predictions[:,1]

    # compute and display the average coverage
    coverage_W2R = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_W2R = np.mean(y_upper - y_lower)
    


   ############### all clean
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf = np.mean(y_upper - y_lower)  




########################### train noisy t 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_t)   
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_t, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_t_train_noisy = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_t_train_noisy = np.mean(y_upper - y_lower)
#    
#    
#    
#    
    
########################### train noisy gaussian 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_gaussian)   
    
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_gaussian, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_gaussian_train_noisy = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_gaussian_train_noisy = np.mean(y_upper - y_lower)
    
    
    
########################### train noisy gumbel 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_gumbel)   
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_gumbel, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_gumbel_train_noisy = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_gumbel_train_noisy = np.mean(y_upper - y_lower)
    
    
    


########################### train noisy positive 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_positive)   
    

    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_positive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_positive_train_noisy_pos = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_positive_train_noisy_pos = np.mean(y_upper - y_lower)
    
    



########################### train noisy center 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_mean)   
    
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_mean, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_mean_train_noisy_mean = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_mean_train_noisy_mean = np.mean(y_upper - y_lower)
    
    
    ###### clean calibration
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_clean_train_noisy_mean = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_clean_train_noisy_mean = np.mean(y_upper - y_lower)
#    
#    
  

########################### train noisy dispersive 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_dispersive)   
    
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_dispersive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_dispersive_train_noisy_dispersive = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_dispersive_train_noisy_dispersive = np.mean(y_upper - y_lower)
    
    
    ###### clean calibration
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage_clean_train_noisy_dispersive = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length_cqr_rf_clean_train_noisy_dispersive = np.mean(y_upper - y_lower)             
             
             
             
             
             
    res = [seed, sigma, coverage, length_cqr_rf, coverage_t, length_cqr_rf_t, coverage_t_train_noisy, length_cqr_rf_t_train_noisy,
          coverage_gaussian, length_cqr_rf_gaussian, coverage_gaussian_train_noisy, length_cqr_rf_gaussian_train_noisy,
          coverage_gumbel, length_cqr_rf_gumbel, coverage_gumbel_train_noisy, length_cqr_rf_gumbel_train_noisy,
          coverage_mean, length_cqr_rf_mean, coverage_mean_train_noisy_mean, length_cqr_rf_mean_train_noisy_mean, coverage_clean_train_noisy_mean, length_cqr_rf_clean_train_noisy_mean,
          coverage_dispersive, length_cqr_rf_dispersive, coverage_dispersive_train_noisy_dispersive, length_cqr_rf_dispersive_train_noisy_dispersive, coverage_clean_train_noisy_dispersive, length_cqr_rf_clean_train_noisy_dispersive,
          coverage_positive, length_cqr_rf_positive, coverage_W2R, length_cqr_rf_W2R,
          coverage_positive_train_noisy_pos, length_cqr_rf_positive_train_noisy_pos]
    col = ['seed', 'sigma', 'coverage', 'length',  'coverage t noise', 'length t', 'coverage t noise, noisy train', 'length t, noisy train',
          'coverage gaussian noise', 'length gaussian', 'coverage gaussian noise, noisy train', 'length gaussian, noisy train',
          'coverage gumbel noise', 'length gumbel', 'coverage gumbel noise, noisy train', 'length gumbel, noisy train',
          'coverage average noise', 'length average', 'coverage average noise, average noisy train', 'length average, average noisy train', 'coverage clean, average noisy train', 'length clean, average noisy train',
          'coverage dispersive noise', 'length dispersive', 'coverage dispersive noise, dispersive noisy train', 'length dispersive, dispersive noisy train', 'coverage clean, dispersive noisy train', 'length clean, dispersive noisy train',
          'coverage positive noise', 'length positive', 'coverage W2R noise', 'length W2R',
          'coverage positive noise, positive noisy train', 'length positive, positive noisy train']
             
             
    results = pd.DataFrame([res], columns=col)
    
    return results
    
  
  
if __name__ == '__main__':
    
  # Parameters
  seed = int(sys.argv[1])
  sigma = float(sys.argv[2])
  # Output directory and filename
  out_dir = "./results_synthetic_reg"
  out_file = out_dir + "_seed_" + str(seed) + "_sigma_" + str(sigma) + ".csv"

  # Run experiment
  result = experiment(seed, sigma)

  # Write the result to output file
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  result.to_csv(out_dir + '/' + out_file, index=False, float_format="%.4f")
  print("Updated summary of results on\n {}".format(out_file))
  sys.stdout.flush()