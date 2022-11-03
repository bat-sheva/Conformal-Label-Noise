import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from dataset.dataset import AVADataset

from model.model import *

sys.path.append('./codes/')
from bbox import train_bbox, predict
from conf_pred import ResCalib, CQR


def experiment(seed):

    #seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    
    # define the paths
    img_path = './data/images'
    train_clean_csv_file = './data/train_labels_avg.csv'
    train_noisy_csv_file = './data/train_labels_noisy.csv'
    test_val_clean_csv_file = './data/test_val_labels_avg.csv'
    test_val_noisy_csv_file = './data/test_val_labels_noisy.csv'

    # training parameters
    decay = True
    conv_base_lr = 5e-5
    dense_lr = 5e-5
    lr_decay_rate = 0.95
    lr_decay_freq = 10
    train_batch_size = 128
    val_batch_size = 1
    test_batch_size = 1
    num_workers = 2 
    epochs = 70
    alpha = 0.1
    optimizer = 'SGD'
    
    
    # misc
    warm_start_epoch = 0
    early_stopping_patience = 10
    save_fig = False
    files_dir = './saved_models_AVA_noisy' 
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)
    file_final_mean = files_dir+'/'+'saved_model_mean_final'
    file_final_qr = files_dir+'/'+'saved_model_qr_final'
    
    num_classes_mean = 1
    num_classes_qr = 2



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    base_model_mean = models.vgg16(pretrained=True)
    bbox_model_mean = NIMA(base_model_mean, num_classes=num_classes_mean, dropout=0.5)
    
    
    base_model_qr = models.vgg16(pretrained=True)
    bbox_model_qr = NIMA(base_model_qr, num_classes=num_classes_qr, dropout=0.2)




    trainset_clean = AVADataset(csv_file=train_clean_csv_file, root_dir=img_path, transform=train_transform)
    trainset_noisy = AVADataset(csv_file=train_noisy_csv_file, root_dir=img_path, transform=train_transform)
       
    val_test_set_clean = AVADataset(csv_file=test_val_clean_csv_file, root_dir=img_path, transform=val_transform)
    val_test_set_noisy = AVADataset(csv_file=test_val_noisy_csv_file, root_dir=img_path, transform=val_transform)

    train_loader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=train_batch_size,
        shuffle=True, num_workers=num_workers)
    train_loader_noisy = torch.utils.data.DataLoader(trainset_noisy, batch_size=train_batch_size,
        shuffle=True, num_workers=num_workers)
        

    if os.path.isfile(file_final_mean):
        print('Loading model instead of training')
        saved_stats = torch.load(file_final_mean, map_location=device)
        bbox_model_mean.load_state_dict(saved_stats['model_state'])
        stats_bbox_mean = saved_stats['stats']
        
    else:
        train_bbox(bbox_model_mean, train_loader_noisy, warm_start_epoch, epochs, conv_base_lr, lr_decay_rate, dense_lr, lr_decay_freq, file_final_mean, num_classes_mean, train_batch_size, optimizer='Adam', decay=decay,
        lr_milestones=10)
        
        
    if os.path.isfile(file_final_qr):
        print('Loading model instead of training')
        saved_stats = torch.load(file_final_qr, map_location=device)
        bbox_model_qr.load_state_dict(saved_stats['model_state'])
        stats_bbox_qr = saved_stats['stats']
        
    else:
        train_bbox(bbox_model_qr, train_loader_noisy, warm_start_epoch, epochs, 1e-3, lr_decay_rate, 1e-3, lr_decay_freq, file_final_qr, num_classes_qr, train_batch_size, optimizer=optimizer, decay=decay, MSE=False,
        lr_milestones=20)    

    
    num_iter = 1
    
    coverage_tmp = np.zeros(num_iter,)
    length_tmp = np.zeros(num_iter,)
    MSE_tmp = np.zeros(num_iter,)
    coverage_val_clean_tmp = np.zeros(num_iter,)
    length_val_clean_tmp = np.zeros(num_iter,)
    cqr_coverage_tmp = np.zeros(num_iter,)
    cqr_length_tmp = np.zeros(num_iter,)
    cqr_coverage_val_clean_tmp = np.zeros(num_iter,)
    cqr_length_val_clean_tmp = np.zeros(num_iter,)
    
    method = np.expand_dims(['Residual scores', 'Residual scores', 'CQR', 'CQR'],1)
    calib_set = np.expand_dims(['noisy', 'clean', 'noisy', 'clean'],1)
    coverage = np.zeros((len(method),1))
    length = np.zeros((len(method),1))
    
    for i in range(num_iter):
    
        random_indices = np.arange(len(val_test_set_clean))
        np.random.shuffle(random_indices)
        split_ind = int(np.floor(0.5 * len(val_test_set_clean)))
        random_indices_val = random_indices[:split_ind]
        random_indices_test = random_indices[split_ind:]
        
        valset_clean = torch.utils.data.Subset(val_test_set_clean, random_indices_val)
        valset_noisy = torch.utils.data.Subset(val_test_set_noisy, random_indices_val)
        testset_clean = torch.utils.data.Subset(val_test_set_clean, random_indices_test)
        testset_noisy = torch.utils.data.Subset(val_test_set_noisy, random_indices_test)
        
        val_loader_clean = torch.utils.data.DataLoader(valset_clean, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers)
        val_loader_noisy = torch.utils.data.DataLoader(valset_noisy, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers)
        test_loader_clean = torch.utils.data.DataLoader(testset_clean, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
        
            
        res_conf_method = ResCalib(bbox_model_mean)
        res_conf_method.calibrate(val_loader_noisy, alpha)
        res_conf_predictions = res_conf_method.predict(test_loader_clean)
        y_lower = res_conf_predictions[:,0]
        y_upper = res_conf_predictions[:,1]
        mean_pred, y_test = predict(bbox_model_mean, test_loader_clean, return_Y=True)
           
        coverage_tmp[i] = np.sum((y_test.squeeze(1) >= y_lower) & (y_test.squeeze(1) <= y_upper)) / len(y_test) * 100
        
        # compute length of the conformal interval per each test point
        length_tmp[i] = np.mean(y_upper - y_lower)
        
        MSE_tmp[i] = mean_squared_error(torch.tensor(mean_pred).cpu().numpy(), y_test.squeeze(1))
        
        
        res_conf_method_val_clean = ResCalib(bbox_model_mean)
        res_conf_method_val_clean.calibrate(val_loader_clean, alpha)
        res_conf_predictions_val_clean = res_conf_method_val_clean.predict(test_loader_clean)
        y_lower_val_clean = res_conf_predictions_val_clean[:,0]
        y_upper_val_clean = res_conf_predictions_val_clean[:,1]
        mean_pred, y_test = predict(bbox_model_mean, test_loader_clean, return_Y=True)
           
        coverage_val_clean_tmp[i] = np.sum((y_test.squeeze(1) >= y_lower_val_clean) & (y_test.squeeze(1) <= y_upper_val_clean)) / len(y_test) * 100
        
        # compute length of the conformal interval per each test point
        length_val_clean_tmp[i] = np.mean(y_upper_val_clean - y_lower_val_clean)
        
        
        
           
        cqr_conf_method = CQR(bbox_model_qr)
        cqr_conf_method.calibrate(val_loader_noisy, alpha)
        cqr_conf_predictions = cqr_conf_method.predict(test_loader_clean)
        cqr_y_lower = cqr_conf_predictions[:,0]
        cqr_y_upper = cqr_conf_predictions[:,1]
        _, y_test = predict(bbox_model_qr, test_loader_clean, return_Y=True)
    #       
        cqr_coverage_tmp[i] = np.sum((y_test.squeeze(1) >= cqr_y_lower) & (y_test.squeeze(1) <= cqr_y_upper)) / len(y_test) * 100
        
        # compute length of the conformal interval per each test point
        cqr_length_tmp[i] = np.mean(cqr_y_upper - cqr_y_lower)
        
        
        cqr_conf_method_val_clean = CQR(bbox_model_qr)
        cqr_conf_method_val_clean.calibrate(val_loader_clean, alpha)
        cqr_conf_predictions_val_clean = cqr_conf_method_val_clean.predict(test_loader_clean)
        cqr_y_lower_val_clean = cqr_conf_predictions_val_clean[:,0]
        cqr_y_upper_val_clean = cqr_conf_predictions_val_clean[:,1]
        _, y_test = predict(bbox_model_qr, test_loader_clean, return_Y=True)
    #       
        cqr_coverage_val_clean_tmp[i] = np.sum((y_test.squeeze(1) >= cqr_y_lower_val_clean) & (y_test.squeeze(1) <= cqr_y_upper_val_clean)) / len(y_test) * 100
        
        # compute length of the conformal interval per each test point
        cqr_length_val_clean_tmp[i] = np.mean(cqr_y_upper_val_clean - cqr_y_lower_val_clean)
    
    coverage[0] = np.mean(coverage_tmp)
    length[0] = np.mean(length_tmp)
    MSE = np.mean(MSE_tmp)
    coverage[1] = np.mean(coverage_val_clean_tmp)
    length[1] = np.mean(length_val_clean_tmp)
    coverage[2] = np.mean(cqr_coverage_tmp)
    length[2] = np.mean(cqr_length_tmp)
    coverage[3] = np.mean(cqr_coverage_val_clean_tmp)
    length[3] = np.mean(cqr_length_val_clean_tmp)
    
    data_array = np.concatenate((coverage, length, method, calib_set), axis=1)
    column_values = ['Coverage', 'Length', 'Method', 'Calib set']
             
    results = pd.DataFrame(data = data_array,  
                  columns = column_values)  
                  
    
    return results
    


if __name__ == '__main__':

  # Parameters
  seed = int(sys.argv[1])

  # Output directory and filename
  out_dir = "./results_AVA_noisy_final"
  out_file = out_dir + "_seed_" + str(seed) + ".csv"

  # Run experiment
  result = experiment(seed)

  # Write the result to output file
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  result.to_csv(out_dir + '/' + out_file, index=False, float_format="%.4f")
  print("Updated summary of results on\n {}".format(out_file))
  sys.stdout.flush()

