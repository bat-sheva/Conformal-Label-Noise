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



def eval_predictions(data_loader, box, data="unknown", plot=False,  printing=True, noise_tr=False):

    if noise_tr:
      Y_pred, Y = box.predict(data_loader, return_y_true = True)
    else:
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


class BboxResnet(nn.Module):

    def __init__(self, base_model, num_classes=10):
        super(BboxResnet, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=base_model.fc.in_features, out_features=num_classes))
            #nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
        
        
# BlackBox method
class BlackBox:

    def __init__(self, num_features, num_classes,
                 family="classification"):

        self.num_features = num_features
        self.num_classes = num_classes
        self.family = family

        self.model = ResNet18()
#        self.model = models.resnet18(pretrained=True)
#        num_ftrs = self.model.fc.in_features
#        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Detect whether CUDA is available
        self.device = device
        self.model = self.model.to(self.device)

        # Define loss functions
        if self.family=="classification":
            self.criterion_pred = nn.CrossEntropyLoss()
        else:
            self.criterion_pred = nn.MSELoss()
                     

    def fit(self, train_loader,
            num_epochs=10, lr=0.001, optimizer = 'Adam', 
            save_model=True, name=None, verbose=False, lr_sch=False):            

        # Process input arguments
        if save_model:
          if name is None:
            raise("Output model name file is needed.")

        # Choose the optimizer
        if optimizer == 'Adam':
          optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'SGD':
          optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Choose the learning rate scheduler
        if lr_sch:
            lr_milestones = [int(num_epochs*0.5)]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)        
        
        # Initialize monitoring variables
        stats = {'epoch': [], "loss": [], 'acc' : []} 

        # Training loop
        print("Begin training.")
        for e in range(1, num_epochs+1):

            epoch_acc = 0
            epoch_loss = 0
            
            self.model.train()

            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                optimizer.zero_grad()

                # Compute model output
                out = self.model(X_batch)
           
                # Samples in group (Z = 0) are processed by the cross-entropy loss
                loss = self.criterion_pred(out, Y_batch)
                
                # Take gradient step
                loss.backward()
                optimizer.step()
                
                # Store information
                acc = accuracy_point(out, Y_batch)
                epoch_acc += acc.item()
                epoch_loss += loss.item()

            epoch_acc /= len(train_loader)
            epoch_loss /= len(train_loader)

            
            if lr_sch:
                scheduler.step()
                                
            stats['epoch'].append(e)
            stats['loss'].append(epoch_loss)
            stats['acc'].append(epoch_acc)


            if verbose:
                print(f'Epoch {e+0:03}: | ', end='')
                print(f'Loss: {epoch_loss:.3f} | ', end='')
                print(f'Acc: {epoch_acc:.3f} | ', end='')
                print('',flush=True)
            
        saved_final_state = dict(stats=stats,
                                 model_state=self.model.state_dict(),
                                 )
        if save_model:
            torch.save(saved_final_state, name)
            
        return stats

    def predict(self, test_loader, return_y_true = False):

        y_pred_list = []
        y_true = []

        with torch.no_grad():
            self.model.eval()
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_true.append(Y_batch.cpu().numpy()[0])
                if self.family=="classification":
                  y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
                  _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                  y_pred_list.append(y_pred_tags.cpu().numpy())
                else:
                  y_pred_list.append(y_test_pred.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        
        if return_y_true:
          return (y_pred, np.array(y_true))
        else:
          return y_pred

    def predict_proba(self, test_loader, return_y_true = False):

        y_proba_list = []
        y_true = []

        with torch.no_grad():
            self.model.eval()
            for X_batch, Y_batch in test_loader:
                y_true.append(Y_batch.cpu().numpy()[0])
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_proba_softmax = torch.softmax(y_test_pred, dim = 1)
                y_proba_list.append(y_proba_softmax.cpu().numpy())
        prob = np.concatenate(y_proba_list)
        prob = prob / prob.sum(axis=1)[:,None]

        if return_y_true:
          return (prob, np.array(y_true))
        else:
          return prob
          
          
          
          
          
          
          

