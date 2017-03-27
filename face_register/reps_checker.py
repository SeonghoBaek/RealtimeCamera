from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
import scipy.stats as stats
import os
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

def drow_distribution(X, labelA=None, labelB=None):
    label_norm = labelA
    label_hist = labelB
    plt.plot(X, mlab.normpdf(X, np.mean(X), np.std(X)), 'o', label=label_norm)
    plt.hist(X, bins=100, normed=True, label=label_hist)
    plt.legend()
    plt.show()

fileDir = os.path.dirname(os.path.realpath(__file__))
embeddingsDir = fileDir + '/output/embedding'
fname = "{}/labels.csv".format(embeddingsDir)
labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]

labels = map(itemgetter(1),
             map(os.path.split,
                 map(os.path.dirname, labels)))  # Get the directory.

fname = "{}/reps.csv".format(embeddingsDir)
embeddings = pd.read_csv(fname, header=None).as_matrix()

first_label = labels[0]

max = 0.0
min = 1.0

total_means = []
class_labels = []
classes = []

embedding_list = []
ems = []

for i in range(len(labels)):
    if labels[i] == first_label:
        #print labels[i], first_label
        classes.append(labels[i])
        ems.append(embeddings[i])

    else:
        #print labels[i], first_label
        class_labels.append(classes)
        embedding_list.append(ems)
        classes = [labels[i]]
        ems = [embeddings[i]]
        first_label = labels[i]

class_labels.append(classes)
embedding_list.append(ems)

#print len(class_labels)

for c in range(len(class_labels)):
    class_mean = []

    for p in range(len(class_labels[c])):
        dist_list = []

        for i in range(len(class_labels[c])):
            if i == p:
                continue

            dst = distance.euclidean(embedding_list[c][i], embedding_list[c][p])
            #dst = distance.euclidean(embeddings[p], embeddings[i])

            if dst > max:
                max = dst

            if dst < min:
                if dst > 0:
                    min = dst

            dist_list.append(dst)

        m = np.mean(dist_list)
        class_mean.append(m)
        #drow_distribution(dist_list, 'norm', 'histogram')

    m = np.mean(class_mean)
    #print class_labels[c][0], m
    total_means.append(m)

print total_means

params = {}
embeddings_dic = {}

for c in range(len(class_labels)):
    params[class_labels[c][0]] = total_means[c]

for c in range(len(class_labels)):
    #print class_labels[c][0]
    embeddings_dic[class_labels[c][0]] = embedding_list[c]

#print len(embeddings_dic['SeonghoBaek'])
#print embeddings_dic['SeonghoBaek'][0]

# Check between different class
total_means = []

for pivot in range(len(class_labels)):
    class_mean = []

    for c in range(len(class_labels)):
        if c == pivot:
            continue

        dist_list = []

        sz = len(class_labels[c])

        if sz > len(class_labels[pivot]):
            sz = len(class_labels[pivot])

        for i in range(sz):
            dst = distance.euclidean(embedding_list[pivot][i], embedding_list[c][i])
            dist_list.append(dst)

        m = np.mean(dist_list)
        class_mean.append(m)

    m = np.mean(class_mean)
    total_means.append(m)

#print total_means

total_means = []

pivot = 5
test_embedding = embedding_list[pivot][0]

class_mean = []

for c in range(len(class_labels)):
    if c == pivot:
        continue

    dist_list = []

    sz = len(class_labels[c])

    for i in range(sz):
        dst = distance.euclidean(test_embedding, embedding_list[c][i])
        dist_list.append(dst)

    m = np.mean(dist_list)
    class_mean.append(m)
    drow_distribution(dist_list, 'norm', 'histogram')

#print class_mean

#drow_distribution(total_means, 'norm', 'histogram')
