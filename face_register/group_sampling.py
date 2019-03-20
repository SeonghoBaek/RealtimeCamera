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


#Grouping user label
#@path: user lable root directory
#@size: group size
def user_grouping(path, size):
    base_label_path = path
    group_label_path = './input/'

    label_dir_list = [d for d in os.listdir(base_label_path) if os.path.isdir(path + d)]

    if size == 0:
        size = 1

    if size > len(label_dir_list):
        size = 1

    group_list = [label_dir_list[i*size:i*size+size] for i in range(1 + len(label_dir_list)/size)]

    #print(group_list)

    if os.path.exists(group_label_path + 'groups'):
        shutil.rmtree(group_label_path + 'groups')

    if os.path.exists(group_label_path + 'groups') == False:
        os.mkdir(group_label_path + 'groups')

    # Create group directories
    for i in range(len(group_list)):
        group_name = 'group' + str(i)

        if os.path.exists(group_label_path + 'groups/' + group_name) == False:
            os.mkdir(group_label_path + 'groups/' + group_name)

        for user_name in group_list[i]:
            user_dir = group_label_path + 'groups/' + group_name + '/' + user_name
            shutil.copytree(base_label_path + user_name, user_dir + '/')

        if os.path.exists(group_label_path + 'groups/' + group_name + '/Unknown') == False:
            os.mkdir(group_label_path + 'groups/' + group_name + '/Unknown')

    #Sample Unknown class for each group
    group_id = 0

    for group in group_list:
        names = [n for n in group]
        unknowns = [u for u in label_dir_list if u not in names]

        print(names)
        print(unknowns)

        print('##########################')
        for unknown in unknowns:
            list_in_unknown = os.listdir(base_label_path + unknown)
            #print(path + unknown)
            #print(list_in_unknown)
            random.shuffle(list_in_unknown)
            #print(list_in_unknown)
            list_in_unknown = random.sample(list_in_unknown, 10)
            #print(list_in_unknown)

            for file_name in list_in_unknown:
                shutil.copy(base_label_path + unknown + '/' + file_name, group_label_path + 'groups/group' + str(group_id) + '/Unknown/')

        group_id = group_id + 1


user_grouping('./input/user/', 5)
