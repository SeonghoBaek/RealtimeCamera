import cv2
import os
import numpy as np


def flip_image(file_name, multiple=False):
    bgrImg = cv2.imread(file_name)

    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(file_name))

    flipped_img = cv2.flip(bgrImg, 1)

    #print bgrImg.shape

    rows = bgrImg.shape[0]
    cols = bgrImg.shape[1]

    M = np.float32([[1, 0, 1], [0, 1, 0]])

    warped_img = cv2.warpAffine(bgrImg, M, (cols, rows))

    cv2.imwrite(file_name + '_f.jpg', flipped_img)
    cv2.imwrite(file_name + '_w.jpg', warped_img)


    #rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)


    # rgbImg = cv2.resize(rgbImg, (0,0), fx=2.0, fy=2.0)



def augment(path):
    label_dir_list = [d for d in os.listdir(path) if d != 'Guest' and d != 'Nobody' and os.path.isdir(path + d)]
    label_list = []

    #print label_dir_list

    for d in label_dir_list:
        listing = [path + d + '/' + f for f in os.listdir(path + d) if f[0] != '.']
        label_list.append(listing)

    #print label_list

    for label_dir in label_list:
        for image_file in label_dir:
            flip_image(image_file)


augment('./input/user/')