import os
import shutil
import random


def guest_sampling(path, size):
    label_dir_list = [d for d in os.listdir(path) if os.path.isdir(path+d) and d != 'User']
    label_dir_size = len(label_dir_list)
    label_list = []
    sample_list = []
    index = 0

    for d in label_dir_list:
        listing = [path + d + '/' + f for f in os.listdir(path + d) if f[0] != '.']
        random.shuffle(listing)
        label_list.append(listing)
        index += 1

    for j in range(size):
        for i in range(label_dir_size):
            if len(label_list[i]) > j:
                sample_list.append(label_list[i][j])

    for i in range(len(sample_list)):
        shutil.copy(sample_list[i], './input/user/Guest/')


def user_sampling(path, size):
    label_dir_list = [d for d in os.listdir(path) if d != 'Guest' and d != 'Nobody' and os.path.isdir(path + d)]
    label_dir_size = len(label_dir_list)
    label_list = []
    sample_list = []
    index = 0

    for d in label_dir_list:
        listing = [path + d + '/' + f for f in os.listdir(path + d) if f[0] != '.']
        random.shuffle(listing)
        label_list.append(listing)
        index += 1

    for j in range(size):
        for i in range(label_dir_size):
            if len(label_list[i]) > j:
                sample_list.append(label_list[i][j])

    if os.path.exists('./input/iguest/User/'):
        shutil.rmtree('./input/iguest/User/')

    os.mkdir('./input/iguest/User')

    for i in range(len(sample_list)):
        shutil.copy(sample_list[i], './input/iguest/User/')


guest_sampling('./input/iguest/', 30)
user_sampling('./input/user/', 15)
