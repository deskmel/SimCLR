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

def eval(checkpoints_folder,data_root,device):
    model = load_model(checkpoints_folder,device)
    X_train,y_train = load_data(data_root)
    X_test,y_test = load_data(data_root,prefix='test')
    X_train_feature = []
    for batch_x, batch_y in next_batch(X_train, y_train, batch_size=config['batch_size']):
        features, _ = model(batch_x)
        X_train_feature.extend(features.cpu().detach().numpy())
    X_train_feature = np.array(X_train_feature)
    X_test_feature = []
    for batch_x, batch_y in next_batch(X_test, y_test, batch_size=config['batch_size']):
        features, _ = model(batch_x)
        X_test_feature.extend(features.cpu().detach().numpy())
    X_test_feature = np.array(X_test_feature)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_feature)

    linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)

def load_data(data_root,prefix='train'):
    X_train = np.fromfile(os.path.join(data_root,'./stl10_binary/' + prefix + '_X.bin'), dtype=np.uint8)
    y_train = np.fromfile(os.path.join(data_root,'./stl10_binary/' + prefix + '_y.bin'), dtype=np.uint8)

    X_train = np.reshape(X_train, (-1, 3, 96, 96)) # CWH
    X_train = np.transpose(X_train, (0, 1, 3, 2)) # CHW

    print("{} images".format(prefix))
    print(X_train.shape)
    print(y_train.shape)
    
    return X_train, y_train - 1
def load_model(checkpoints_folder,device):
    model =ResNetSimCLR(**config['model'])
    model.eval()
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model
def next_batch(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        X_batch = torch.tensor(X[i: i+batch_size]) / 255.
        y_batch = torch.tensor(y[i: i+batch_size])
        yield X_batch.to(device), y_batch.to(device)
    
def linear_model_eval(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    print("Logistic Regression feature eval")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))
    
    print("-------------------------------")
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    print("KNN feature eval")
    print("Train score:", neigh.score(X_train, y_train))
    print("Test score:", neigh.score(X_test, y_test))
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    folder_name = '/data2/meng/SimCLR/SimCLRpytorch/SimCLR/runs/May17_10-41-45_IEEE-8889-1080Ti'
    checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"))
    data_root = '/data2/meng/SimCLR/SimCLRpytorch/SimCLR/data/'
    eval(checkpoints_folder,data_root,device)