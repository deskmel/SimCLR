import torch
import sys
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from models.resnet_simclr import ResNetSimCLR
import importlib.util
import torchvision
import torchvision.transforms as transforms

def eval(model,data_root,device,config):
    train_loader,test_loader = load_dataset(data_root,config)
    X_train_feature = []
    y_train = []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_train_feature.extend(features.cpu().detach().numpy())
        y_train.extend(batch_y.cpu().detach().numpy())
    X_train_feature = np.array(X_train_feature)
    X_test_feature = []
    y_train = np.array(y_train)
    y_test = []
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        features, _ = model(batch_x)
        X_test_feature.extend(features.cpu().detach().numpy())
        y_test.extend(batch_y.cpu().detach().numpy())
    X_test_feature = np.array(X_test_feature)
    y_test = np.array(y_test)
    scaler = preprocessing.StandardScaler()
    print('ok')
    scaler.fit(X_train_feature)
    #print(X_test_feature.shape)
    #print(y_test.shape)
    linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)



def load_dataset(root = './data',config):
    train_dataset = torchvision.datasets.CIFAR10(
        root,
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        train=True,
        download=True,
        )
    test_dataset = torchvision.datasets.CIFAR10(
        root,
        transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        train=False,
        download=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    return train_loader,test_loader
def load_model(checkpoints_folder,device):
    model =ResNetSimCLR(**config['model'])
    model.eval()
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

    
def linear_model_eval(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    print("Logistic Regression feature eval")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))
    '''
    print("-------------------------------")
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    print("KNN feature eval")
    print("Train score:", neigh.score(X_train, y_train))
    print("Test score:", neigh.score(X_test, y_test))
    '''
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    folder_name = './runs/May17_20-55-07_userme'
    checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
    data_root = './data/'
    model = load_model(checkpoints_folder,device)
    eval(model,data_root,device,config)
    
    #load_dataset()