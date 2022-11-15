import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import os 
from resnet import ResNet18
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)
        
        
def accuracy_point(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc*100


def predict(model, test_loader, return_y_true = False):

    y_pred_list = []
    y_true = []
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_true.append(Y_batch.cpu().numpy()[0])
            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred = np.concatenate(y_pred_list)
    
    if return_y_true:
      return (y_pred, np.array(y_true))
    else:
      return y_pred

def predict_proba(model, test_loader, return_y_true = False):

    y_proba_list = []
    y_true = []
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for X_batch, Y_batch in test_loader:
            y_true.append(Y_batch.cpu().numpy()[0])
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_proba_softmax = torch.softmax(y_test_pred, dim = 1)
            y_proba_list.append(y_proba_softmax.cpu().numpy())
    prob = np.concatenate(y_proba_list)
    prob = prob / prob.sum(axis=1)[:,None]

    if return_y_true:
      return (prob, np.array(y_true))
    else:
      return prob



def eval_predictions(data_loader, box, data="unknown", plot=False,  printing=True):


    Y_pred, Y = predict(box, data_loader, return_y_true = True)
    
    if plot:
        A = confusion_matrix(Y, Y_pred)
        df_cm = pd.DataFrame(A, index = [i for i in range(K)], columns = [i for i in range(K)])
        plt.figure(figsize = (4,3))
        pal = sns.light_palette("navy", as_cmap=True)
        sn.heatmap(df_cm, cmap=pal, annot=True, fmt='g')

    class_acc = np.mean(Y==Y_pred)
    if printing:
        print("Classification acc on {:s} data: {:.1f}%".format(data, class_acc*100))
    return (class_acc*100)


          
          
          
          
          
          
          

