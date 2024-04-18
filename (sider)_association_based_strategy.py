
import dgl
import torch
import random
import cv2
import torchvision
import pandas as pd
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

from dgllife.data import SIDER
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import  History
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm, trange
from sklearn.model_selection import train_test_split

from utils.general import DATASET, get_dataset, separate_active_and_inactive_data
from utils.general import  get_embedding_vector_class, count_lablel,data_generator
from utils.gcn_pre_trained import get_sider_model

from Models.heterogeneous_siamese_sider import siamese_model_attentiveFp_sider, siamese_model_Canonical_sider


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Data"""

cache_path='./sider_dglgraph.bin'

df = get_dataset("sider")

sider_tasks = df.columns.values[1:].tolist()

from collections import Counter

one = []
zero = []

for task in sider_tasks:
  data = df[task]
  print(task ,":" ,Counter(data))
  zero.append(Counter(data)[0])
  one.append(Counter(data)[1])

sum(one), sum(zero)

# Importing the matplotlib library
import numpy as np
import matplotlib.pyplot as plt
# Declaring the figure or the plot (y, x) or (width, height)
plt.figure(figsize=[20, 15])
# Data to be plotted
X = np.arange(1,len(sider_tasks)+1)

plt.bar(X + 0.20, zero, color = 'g', width = 0.25)
plt.bar(X + 0.4, one, color = 'b', width = 0.25)
# Creating the legend of the bars in the plot
plt.legend(['Active' , 'inactive' ])
# Overiding the x axis with the country names
plt.xticks([i + 0.25 for i in range(1,28)], X)
# Giving the tilte for the plot
plt.title("Sider dataset diagram")
# Namimg the x and y axis
plt.xlabel('sider_tasks')
plt.ylabel('Cases')
# Saving the plot as a 'png'
plt.savefig('4BarPlot.png')
# Displaying the bar plot
plt.show()

"""# Required functions"""

from dgllife.model import MLPPredictor

def create_dataset_with_gcn(dataset, class_embed_vector, GCN, tasks ):
    created_data = []
    data = np.arange(len(tasks))
    onehot_encoded = to_categorical(data)
    for i, data in enumerate(dataset):
        smiles, g, labels, mask = data
        g = g.to(device)
        g = dgl.add_self_loop(g)
        graph_feats = g.ndata.pop('h')
        embbed = GCN(g, graph_feats)
        embbed = embbed.to('cpu')
        embbed = embbed.detach().numpy()
        for j, label in enumerate(labels):
            a = (embbed, onehot_encoded[j], class_embed_vector[j], label, j , tasks[j])
            created_data.append(a)
    print('Data created!!')
    return created_data

"""# Calculation of embedded vectors for each class"""

df_positive, df_negative = Separate_active_and_inactive_data(df, sider_tasks)

for i,d in enumerate(zip(df_positive,df_negative)):
    print(f'{sider_tasks[i]}=> positive: {len(d[0])} - negative: {len(d[1])}')

dataset_positive = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path) for d in df_positive]
dataset_negative = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path) for d in df_negative]

embed_class_sider = get_embedding_vector_class(dataset_positive, dataset_negative, radius=2, size = 512)

"""# Classification with BioAct-Het and AttentiveFp GCN"""

model_name = 'GCN_attentivefp_SIDER'
gcn_model = get_sider_model(model_name)
gcn_model.eval()
gcn_model = gcn_model.to(device)

dataset = DATASET(df,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path)
data_ds = create_dataset_with_gcn(dataset, embed_class_sider, gcn_model, sider_tasks)

from sklearn.model_selection import KFold

Epoch_S = 10

def evaluate_model(df, k = 10 , shuffle = False):

    result = []
    s = 0

    kf = KFold(n_splits=10, shuffle= shuffle, random_state=None)

    for train_index, test_index in kf.split(df):

        train_ds = [df[index] for index in train_index]

        valid_ds = [df[index] for index in test_index]

        label_pos , label_neg = count_lablel(train_ds)
        print(f'train positive label: {label_pos} - train negative label: {label_neg}')

        label_pos , label_neg = count_lablel(valid_ds)
        print(f'Test positive label: {label_pos} - Test negative label: {label_neg}')

        l_train = []
        r_train = []
        lbls_train = []
        l_valid = []
        r_valid = []
        lbls_valid = []

        for i , data in enumerate(train_ds):
            embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_train.append(embbed_drug[0])
            r_train.append(embbed_task)
            lbls_train.append(lbl.tolist())

        for i , data in enumerate(valid_ds):
            embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_valid.append(embbed_drug[0])
            r_valid.append(embbed_task)
            lbls_valid.append(lbl.tolist())

        l_train = np.array(l_train).reshape(-1,1024,1)
        r_train = np.array(r_train).reshape(-1,512,1)
        lbls_train = np.array(lbls_train)

        l_valid = np.array(l_valid).reshape(-1,1024,1)
        r_valid = np.array(r_valid).reshape(-1,512,1)
        lbls_valid = np.array(lbls_valid)

        # create neural network model
        siamese_net = siamese_model_attentiveFp_sider()
        history = History()
        P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size = 128, callbacks=[history])

        for j in range(100):
            C=1
            Before = int(P.history['accuracy'][-1]*100)
            for i in range(2,Epoch_S+1):
                if  int(P.history['accuracy'][-i]*100) == Before:
                    C=C+1
                else:
                    C=1
                Before=int(P.history['accuracy'][-i]*100)
                print(Before)
            if C==Epoch_S:
                break
            P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size = 128, callbacks=[history])
        print(j+1)

        score  = siamese_net.evaluate([l_valid,r_valid], lbls_valid, verbose=1)
        a = (score[1],score[4])
        result.append(a)

        if score[4] > s :
            best_model = siamese_net
            s = score[4]
            print("Save_model")


    return result , best_model

scores, best_model = evaluate_model(data_ds, 10, True)

l_train = []
r_train = []
lbls_train = []
task_number_train = []

l_valid = []
r_valid = []
lbls_valid = []
task_namber_test = []

for i , data in enumerate(train_ds):
    embbed_drug, onehot_task, embbed_task, lbl, task_number, task_name = data
    if lbl == 1.:
        l_train.append(embbed_drug[0])
        r_train.append(embbed_task)
        lbls_train.append(lbl.tolist())
        task_number_train.append(task_number)


for i, data in enumerate(valid_ds):
    embbed_drug, onehot_task, embbed_task, lbl, task_number, task_name = data
    if lbl == 1.:
        l_valid.append(embbed_drug[0])
        r_valid.append(embbed_task)
        lbls_valid.append(lbl.tolist())
        task_namber_test.append(task_number)

l_train = np.array(l_train).reshape(-1,1024,1)
r_train = np.array(r_train).reshape(-1,512,1)
lbls_train = np.array(lbls_train)

l_valid = np.array(l_valid).reshape(-1,1024,1)
r_valid = np.array(r_valid).reshape(-1,512,1)
lbls_valid = np.array(lbls_valid)

model = siamese_model()

# Train the siamese model with your training data

# Create a new Keras model that outputs the output of the L1 layer in the siamese model
L1_model = tf.keras.Model(inputs=model.inputs,
                          outputs=model.layers[5].output)

# Get the output of the L1 layer for the test data
before_training_L1_output_valid = L1_model.predict([l_valid, r_valid])
before_training_L1_output_train = L1_model.predict([l_train, r_train])

# Train the siamese model with your training data

# Create a new Keras model that outputs the output of the L1 layer in the siamese model
L1_model = tf.keras.Model(inputs=best_model.inputs,
                          outputs=best_model.layers[5].output)

# Get the output of the L1 layer for the test data
after_training_L1_output_valid = L1_model.predict([l_valid, r_valid])
after_training_L1_output_train = L1_model.predict([l_train, r_train])

"""### T-SNE Visualization n_components = 3"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = before_training_L1_output_train.copy()  # Replace this with your actual data
labels = task_number_train.copy()

# Define the t-SNE model with default parameters
tsne = TSNE(n_components=3, init='random', learning_rate=200.0, verbose=1)

# Fit and transform the data to 2 dimensions
tsne_result = tsne.fit_transform(data)

# Plot the t-SNE results
plt.figure(figsize=(10, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='jet', s=5)
plt.colorbar()
plt.title('t-SNE Visualization Train Data Before Training')
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = after_training_L1_output_train.copy()  # Replace this with your actual data
labels = task_number_train.copy()

# Define the t-SNE model with default parameters
tsne = TSNE(n_components=3, init='random', learning_rate=200.0, verbose=1)

# Fit and transform the data to 2 dimensions
tsne_result = tsne.fit_transform(data)

# Plot the t-SNE results
plt.figure(figsize=(10, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='jet', s=5)
plt.colorbar()
plt.title('t-SNE Visualization Train Data After Training')
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = before_training_L1_output_valid.copy()  # Replace this with your actual data
labels = task_namber_test.copy()

# Define the t-SNE model with default parameters
tsne = TSNE(n_components=3, init='random', learning_rate=200.0, verbose=1)

# Fit and transform the data to 2 dimensions
tsne_result = tsne.fit_transform(data)

# Plot the t-SNE results
plt.figure(figsize=(10, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='jet', s=5)
plt.colorbar()
plt.title('t-SNE Visualization Test Data Before Training')
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = after_training_L1_output_valid.copy()  # Replace this with your actual data
labels = task_namber_test.copy()

# Define the t-SNE model with default parameters
tsne = TSNE(n_components=2, init='random', learning_rate=200.0, verbose=1)

# Fit and transform the data to 2 dimensions
tsne_result = tsne.fit_transform(data)

# Plot the t-SNE results
plt.figure(figsize=(10, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='jet', s=5)
plt.colorbar()
plt.title('t-SNE Visualization Test Data After Training')
plt.show()

"""### radius = 2, size = 512,  epochs = 10, bach = 128"""

scores

acc = []
auc = []
for i in scores:
    acc.append(i[0])
    auc.append(i[1])
print(f'accuracy= {np.mean(acc)} AUC= {np.mean(auc)}')

"""# Classification with BioAct-Het and Canonical GCN"""

model_name = 'GCN_canonical_SIDER'
gcn_model = get_sider_model(model_name)
gcn_model.eval()
gcn_model = gcn_model.to(device)

dataset = DATASET(df,smiles_to_bigraph, CanonicalAtomFeaturizer(), cache_file_path = cache_path)
data_ds = create_dataset_with_gcn(dataset, embed_class_sider, gcn_model, sider_tasks)

from sklearn.model_selection import KFold

Epoch_S = 10

def evaluate_model(df, k = 10 , shuffle = False):
    result =[]

    kf = KFold(n_splits=10, shuffle= shuffle, random_state=None)

    for train_index, test_index in kf.split(df):

        train_ds = [df[index] for index in train_index]

        valid_ds = [df[index] for index in test_index]

        label_pos , label_neg = count_lablel(train_ds)
        print(f'train positive label: {label_pos} - train negative label: {label_neg}')

        label_pos , label_neg = count_lablel(valid_ds)
        print(f'Test positive label: {label_pos} - Test negative label: {label_neg}')

        l_train = []
        r_train = []
        lbls_train = []
        l_valid = []
        r_valid = []
        lbls_valid = []

        for i , data in enumerate(train_ds):
            embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_train.append(embbed_drug[0])
            r_train.append(embbed_task)
            lbls_train.append(lbl.tolist())

        for i , data in enumerate(valid_ds):
            embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_valid.append(embbed_drug[0])
            r_valid.append(embbed_task)
            lbls_valid.append(lbl.tolist())

        l_train = np.array(l_train).reshape(-1,512,1)
        r_train = np.array(r_train).reshape(-1,512,1)
        lbls_train = np.array(lbls_train)

        l_valid = np.array(l_valid).reshape(-1,512,1)
        r_valid = np.array(r_valid).reshape(-1,512,1)
        lbls_valid = np.array(lbls_valid)

        # create neural network model
        siamese_net = siamese_model_Canonical_sider()
        history = History()
        P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size=64, callbacks=[history])

        for j in range(100):
            C=1
            Before = int(P.history['accuracy'][-1]*100)
            for i in range(2,Epoch_S+1):
                if  int(P.history['accuracy'][-i]*100)== Before:
                    C=C+1
                else:
                    C=1
                Before=int(P.history['accuracy'][-i]*100)
                print(Before)
            if C==Epoch_S:
                break
            P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size=64, callbacks=[history])

        print(j+1)

        score  = siamese_net.evaluate([l_valid,r_valid],lbls_valid, verbose=1)
        a = (score[1],score[4])
        result.append(a)

    return result


scores = evaluate_model(data_ds, 10, True)

scores

acc = []
auc = []
for i in scores:
    acc.append(i[0])
    auc.append(i[1])
print(f'accuracy= {np.mean(acc)} AUC= {np.mean(auc)}')