import os
import sys
import torch
import random
import pandas as pd
import numpy as np

sys.path.append('./codes/')

from cqr import helper
from CQR import CQR_rf
from Data_Generation_Model import f
from reg_noise import add_Y_noise_additive, add_Y_noise_dependant


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
    y_train_noisy_gaussian = add_Y_noise_additive(y_train, sigma=sigma, gaussian=True)
    y_calib_noisy_gaussian = add_Y_noise_additive(y_calib, sigma=sigma, gaussian=True)
    y_test_noisy_gaussian = add_Y_noise_additive(y_test, sigma=sigma, gaussian=True)
    
    y_train_noisy_t = add_Y_noise_additive(y_train, sigma=sigma)
    y_calib_noisy_t = add_Y_noise_additive(y_calib, sigma=sigma)
    y_test_noisy_t = add_Y_noise_additive(y_test, sigma=sigma)
    
    y_train_noisy_gumbel = add_Y_noise_additive(y_train, sigma=sigma, gumbel=True)
    y_calib_noisy_gumbel = add_Y_noise_additive(y_calib, sigma=sigma, gumbel=True)
    y_test_noisy_gumbel = add_Y_noise_additive(y_test, sigma=sigma, gumbel=True)
    
    y_train_noisy_positive = add_Y_noise_additive(y_train, positive=True, sigma=sigma)
    y_calib_noisy_positive = add_Y_noise_additive(y_calib, positive=True, sigma=sigma)
    y_test_noisy_positive = add_Y_noise_additive(y_test, positive=True, sigma=sigma)
    
    
    y_train_noisy_contractive = add_Y_noise_dependant(y_train)
    y_calib_noisy_contractive = add_Y_noise_dependant(y_calib)
    y_test_noisy_contractive = add_Y_noise_dependant(y_test)
    
    
    y_train_noisy_dispersive = add_Y_noise_dependant(y_train, contractive=False)
    y_calib_noisy_dispersive = add_Y_noise_dependant(y_calib, contractive=False)
    y_test_noisy_dispersive = add_Y_noise_dependant(y_test, contractive=False)
    
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
    
    Noise_type_train = np.expand_dims(['clean', 'clean', 'clean', 'clean', 'clean', 'clean', 'clean', 'clean', 't', 'gaussian', 'gumbel', 'positive', 'contractive', 'contractive', 'dispersive', 'dispersive'],1)
    Noise_type_calib = np.expand_dims(['t', 'gaussian', 'gumbel', 'positive', 'contractive', 'dispersive', 'wrong_to_right', 'clean', 't', 'gaussian', 'gumbel', 'positive', 'contractive', 'clean', 'dispersive', 'clean'],1)
    calib = np.expand_dims(['noisy', 'noisy', 'noisy', 'noisy', 'noisy', 'noisy', 'noisy', 'clean', 'noisy', 'noisy', 'noisy', 'noisy', 'noisy', 'clean', 'noisy', 'clean'],1)
    coverage = np.zeros((len(Noise_type_train),1))
    seed = np.ones((len(Noise_type_train),1))*seed
    sigma = np.ones((len(Noise_type_train),1))*sigma
    length = np.zeros((len(Noise_type_train),1))
    
    
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
    coverage[0] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[0] = np.mean(y_upper - y_lower)
    
    
    
    
    ############### gaussian noise
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_gaussian, alpha)
    predictions = cqr_conf_method.predict(x_test)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[1] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[1] = np.mean(y_upper - y_lower)
    
    
    
    
    ############### gumbel noise
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_gumbel, alpha)
    predictions = cqr_conf_method.predict(x_test)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[2] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[2] = np.mean(y_upper - y_lower)
    
    
    
    
    
   ############### positive noise
    
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_positive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[3] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[3] = np.mean(y_upper - y_lower)
    
    
    
   ############### contractive noise
    
   
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_contractive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[4] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[4] = np.mean(y_upper - y_lower)
    
    
    
    ############## dispersive noise
    
   
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_dispersive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[5] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[5] = np.mean(y_upper - y_lower)
    
    
    
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
    coverage[6] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[6] = np.mean(y_upper - y_lower)
    


   ############### all clean
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib, alpha)
    predictions = cqr_conf_method.predict(x_test)
    
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[7] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[7] = np.mean(y_upper - y_lower)  




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
    coverage[8] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[8] = np.mean(y_upper - y_lower)
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
    coverage[9] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[9] = np.mean(y_upper - y_lower)
    
    
    
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
    coverage[10] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[10] = np.mean(y_upper - y_lower)
    
    
    


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
    coverage[11]= np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[11] = np.mean(y_upper - y_lower)
    
    



########################### train noisy contractive 

    # define the QRF model
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                               fit_params=None,
                                                               quantiles=[5, 95],
                                                               params=params_qforest)
    
    quantile_estimator.fit(x_train, y_train_noisy_contractive)   
    
    
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib_noisy_contractive, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[12] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[12] = np.mean(y_upper - y_lower)
    
    
    ###### clean calibration
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[13] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[13] = np.mean(y_upper - y_lower)
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
    coverage[14] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[14] = np.mean(y_upper - y_lower)
    
    
    ###### clean calibration
    
    cqr_conf_method = CQR_rf(quantile_estimator)
    cqr_conf_method.calibrate(x_calib, y_calib, alpha)
    predictions = cqr_conf_method.predict(x_test)
    

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute and display the average coverage
    coverage[15] = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100
    
    # compute length of the conformal interval per each test point
    length[15] = np.mean(y_upper - y_lower)             
             
    data_array = np.concatenate((coverage, length, Noise_type_train, Noise_type_calib, calib, sigma, seed), axis=1)
    column_values = ['coverage', 'length', 'Noise type train', 'Noise type calib', 'calib', 'sigma', 'seed']
             
    results = pd.DataFrame(data = data_array,  
                  columns = column_values)         
             
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