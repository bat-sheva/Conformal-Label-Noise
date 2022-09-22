import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
import pandas as pd
import random
from sklearn.model_selection import train_test_split


sys.path.append('./codes/')

from split_conf import SplitConformal, evaluate_predictions
from black_boxes_CNN import BlackBox, ClassifierDataset, eval_predictions, predict, predict_proba
from cifar10_models.resnet import resnet18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
def experiment(seed):

    #seed = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    noise_tr = False
        
    files_dir = './saved_models_cifar10' 
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    if noise_tr:

      
      augmentation = [
  #            transforms.RandomCrop(32, 4),
  #            transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                   std = [0.2471, 0.2435, 0.2616])
  #            transforms.Normalize(mean=[0.485, 0.456, 0.406],
  #                                     std=[0.229, 0.224, 0.225])
          ]

    else:
    
      augmentation = [
  #            transforms.RandomCrop(32, 4),
  #            transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                   std = [0.2471, 0.2435, 0.2616])
  #            transforms.Normalize(mean=[0.485, 0.456, 0.406],
  #                                     std=[0.229, 0.224, 0.225])
          ]
    transform = transforms.Compose(augmentation)
    
    test_dataset_tmp = torchvision.datasets.CIFAR10(root='./cifar_10', train=False, download=False, transform=transform)
    test_loader_tmp = torch.utils.data.DataLoader(test_dataset_tmp, batch_size=10000, shuffle=False, num_workers=2, drop_last=True)
    
    for test_images, _ in test_loader_tmp:
        test_images_array = test_images
        
    cifar10h_raw = pd.read_csv(r'./cifar10h-raw.csv')
    cifar10h_raw.head()

    cifar10_test_idx = np.asarray(cifar10h_raw.iloc[0:,8]).astype(int)
    _, unique_indices = np.unique(cifar10_test_idx, return_index=True)
    unique_indices = unique_indices[1:]
    cifar10h_unique = cifar10h_raw.iloc[unique_indices,:]
    noisy_y = np.asarray(cifar10h_unique.iloc[:,6]).astype(int)
    clean_y = np.asarray(cifar10h_unique.iloc[:,5]).astype(int)
    noisy_clean_y = np.concatenate((np.expand_dims(noisy_y,1),np.expand_dims(clean_y,1)),axis=1)
    
    
    
    
    if noise_tr:    
    
      X_train_tmp, X_test, Y_train_noisy_clean_tmp, Y_test_noisy_clean = train_test_split(test_images_array, noisy_clean_y, test_size=0.2, random_state=seed) 
      Y_test_noisy = Y_test_noisy_clean[:,0]
      Y_test_clean = Y_test_noisy_clean[:,1]
      Y_train_noisy_tmp = Y_train_noisy_clean_tmp[:,0]
   
      X_train, X_calib, Y_train_noisy, Y_calib_noisy = train_test_split(X_train_tmp, Y_train_noisy_tmp, test_size=0.2, random_state=seed)
      
      
      train_dataset = ClassifierDataset(X_train, torch.from_numpy(Y_train_noisy).long())
      train_loader = DataLoader(train_dataset, batch_size=128, 
                                shuffle=True, drop_last=True)
                              
                              
                              
    else:                         

      X_calib, X_test, Y_calib_noisy_clean, Y_test_noisy_clean = train_test_split(test_images_array, noisy_clean_y, test_size=0.8, random_state=seed) 
      Y_test_noisy = Y_test_noisy_clean[:,0]
      Y_test_clean = Y_test_noisy_clean[:,1]
      Y_calib_noisy = Y_calib_noisy_clean[:,0]
      Y_calib_clean = Y_calib_noisy_clean[:,1]
                              
    calib_dataset = ClassifierDataset(X_calib, torch.from_numpy(Y_calib_noisy).long())
    calib_loader = DataLoader(calib_dataset, batch_size=1, 
                              shuffle=False)
                              
    if noise_tr==False:
      calib_dataset_clean = ClassifierDataset(X_calib, torch.from_numpy(Y_calib_clean).long())
      calib_loader_clean = DataLoader(calib_dataset_clean, batch_size=1, 
                                      shuffle=False)
                              
    test_noisy_dataset = ClassifierDataset(X_test, torch.from_numpy(Y_test_noisy).long())
    test_noisy_loader = DataLoader(test_noisy_dataset, batch_size=1, 
                                    shuffle=False)
                                    
    test_clean_dataset = ClassifierDataset(X_test, torch.from_numpy(Y_test_clean).long())
    test_clean_loader = DataLoader(test_clean_dataset, batch_size=1, 
                                    shuffle=False)
                          
                          
    out_prefix =  'seed_%a' %seed

    
    # Train model with CE loss
    
    if noise_tr:  
    
      #bbox = models.resnet18(pretrained=True)
           
      file_final = files_dir+'/'+'saved_model_' + out_prefix
        
      if os.path.isfile(file_final):
          print('Loading model instead of training')
          bbox = BlackBox(num_features=3, 
                                          num_classes=10)
          saved_stats = torch.load(file_final, map_location=device)
          bbox.model.load_state_dict(saved_stats['model_state'])
          stats_bbox = saved_stats['stats']
  
      else:
          bbox = BlackBox(num_features=3, 
                                  num_classes=10)
          stats_bbox = bbox.fit(train_loader = train_loader, 
                                  num_epochs = 500, 
                                  optimizer = 'SGD',
                                  lr = 0.1,
                                  save_model = True,
                                  lr_sch = True,
                                  name = file_final,
                                  verbose=True)
                                  
    else:
                  
      bbox = resnet18(pretrained=True)
    
    alpha = 0.1

    sc_method = SplitConformal()
    sc_method.calibrate(calib_loader, alpha, bbox, noise_tr=noise_tr)

    
    sets_noisy = sc_method.predict(test_noisy_loader)
    if noise_tr:
      _, y_noisy_true = bbox.predict(test_noisy_loader, return_y_true = True)
    else:
      _, y_noisy_true = predict(bbox, test_noisy_loader, return_y_true = True)
    res_noisy = evaluate_predictions(sets_noisy, y_noisy_true, noisy=True)
    res_noisy['Acc_noisy'] = eval_predictions(test_noisy_loader, bbox, data="test", plot=False, printing=False, noise_tr=noise_tr)
    
    sets_clean = sc_method.predict(test_clean_loader)
    if noise_tr:
      _, y_clean_true = bbox.predict(test_clean_loader, return_y_true = True)
    else:
      _, y_clean_true = predict(bbox, test_clean_loader, return_y_true = True)
    res_clean = evaluate_predictions(sets_clean, y_clean_true, clean=True)
    res_clean['Acc_clean'] = eval_predictions(test_clean_loader, bbox, data="test", plot=False, printing=False, noise_tr=noise_tr)
    
    if noise_tr:
      results = pd.concat([res_noisy, res_clean], axis=1, join="inner")
    
    else:   
      sc_method_calib_clean = SplitConformal()
      sc_method_calib_clean.calibrate(calib_loader_clean, alpha, bbox)
      sets_calib_clean = sc_method_calib_clean.predict(test_clean_loader)
      _, y_clean_true = predict(bbox, test_clean_loader, return_y_true = True)
      res_calib_clean = evaluate_predictions(sets_calib_clean, y_clean_true)
      res_calib_clean['Acc'] = eval_predictions(test_clean_loader, bbox, data="test", plot=False, printing=False)
  
      results = pd.concat([res_noisy, res_clean, res_calib_clean], axis=1, join="inner")
    
#    
        
    results = results.reset_index()
    return results
    
    

  
  
  
if __name__ == '__main__':
    
  # Parameters
  seed = int(sys.argv[1])
  # Output directory and filename
  out_dir = "./results_cifar10"
  out_file = out_dir + "_seed_" + str(seed) + ".csv"

  # Run experiment
  result = experiment(seed)

  # Write the result to output file
  if not os.path.exists(out_dir):
      os.mkdir(out_dir)
  result.to_csv(out_dir + '/' + out_file, index=False, float_format="%.4f")
  print("Updated summary of results on\n {}".format(out_file))
  sys.stdout.flush()