import os
import cv2
import copy
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import rotate
from skimage.filters import gaussian
from skimage.util import random_noise

class DataLoader:
    def __init__(self, path, sz, filename):
        self.path = path
        self.df = self.load_data("driving_log.csv")
        self.new_df = self.load_data(filename)
        self.h, self.w, self.ch = sz

    def load_data(self, file_name):
        path_to_csv = os.path.join(self.path, file_name)
        df = pd.read_csv(path_to_csv)
        return df

    def load_image(self, file_name):
        path_to_image = os.path.join(self.path, file_name)
        image = cv2.imread(path_to_image)
        return image

    def match_X_y(self):
        new_df = pd.DataFrame(columns=['image', 'steering_angle'])
        new_df['image'] = None
        new_df['steering_angle'] = 0.
        cols = ['center', 'left', 'right']
        k = 0
        for col in cols:
            for i in range(len(self.df)):
                new_df.at[i+k, 'image'] = self.df.at[i, col]
                if col=='center':
                    new_df.at[i+k, 'steering_angle'] = self.df.at[i, 'steering']
                elif col=='left':
                    new_df.at[i+k, 'steering_angle'] = self.df.at[i, 'steering'] + 0.25
                else:
                    new_df.at[i+k, 'steering_angle'] = self.df.at[i, 'steering'] - 0.25
            k += len(self.df)
        new_df.to_csv(os.path.join(self.path, "extended_log.csv"))

    def flip_horizontal(self, X_i):
        image = X_i[:, ::-1]
        return image

    #reference from https://subscription.packtpub.com/book/application_development/9781785283932/1/ch01lvl1sec11/image-translation
    def shift(self, X_i, y_i):
        w = X_i.shape[1]
        h = X_i.shape[0]
        #shift range between -0.10 and 0.10
        shift_w = w * (random.random()*0.2-0.1)
        shift_h = h * (random.random()*0.2-0.1)
        #adjust steering angle slightly
        s_angle = y_i + shift_w*1e-04
        #M = 2x3 transformation matrix
        M = np.float32([[1, 0, shift_w],[0, 1, shift_h]])
        #https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=getrotationmatrix2d
        image = cv2.warpAffine(X_i, M, (w, h))
        return image, s_angle

    def blur(self, X_i):
        random_int = random.randrange(4)
        image = gaussian(X_i, random_int, multichannel=True, mode='reflect')
        return image

    #reference from https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    def adjust_brightness(self, X_i):
        #change to hsv
        img_hsv = cv2.cvtColor(X_i, cv2.COLOR_RGB2HSV)
        #brigthness, third channel - value range is [0,255], can be darker or brighter by degree between -0.5 and 0.5
        brightness = img_hsv[:,:,2]*(random.random()-0.5)
        img_hsv[:,:,2] += brightness.astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    def add_noise(self, X_i):
        image = random_noise(X_i, mode='gaussian')
        return image

    def split_data(self):
        if self.new_df.shape[0]:
            #split train and valid set
            X_train, X_valid, y_train, y_valid = train_test_split(self.new_df['image'], self.new_df['steering_angle'],
                                                                  test_size=0.1,
                                                                  random_state=42, shuffle=True)
        return X_train, X_valid, y_train, y_valid

    def augment_data(self, X, y):
        if len(X):
            X = X.tolist()
            y = y.tolist()
            X_augmented = []
            y_augmented = []
            for i in range(len(X)):
                X_i = self.load_image(X[i].strip())
                X_copy = copy.deepcopy(X_i)
                y_copy = copy.deepcopy(y[i])

                #random probability and threshold
                rand_prob = random.random()
                thres = 0.3
                if rand_prob > thres:
                    rand_transform = random.randrange(5)
                    if rand_transform==0:
                        X_copy = self.flip_horizontal(X_copy)
                        y_copy = -y_copy
                    if rand_transform==1:
                        X_copy, y_copy = self.shift(X_copy, y_copy)
                    if rand_transform==2:
                        X_copy = self.blur(X_copy)
                    if rand_transform==3:
                        X_copy = self.adjust_brightness(X_copy)
                    if rand_transform==4:
                        X_copy = self.add_noise(X_copy)
                X_copy = X_copy[50:-20, :, :]
                X_augmented.append(X_copy)
                y_augmented.append(y_copy)
        assert len(X_augmented)==len(y_augmented)
        return X_augmented, y_augmented


def generate_batch(X, y, bs, data):
    len_X = len(X)
    #for generator, a continuous loop
    while True:
        X, y = shuffle(X, y)
        epo = 0
        #batch generator
        for offset in range(0, len_X*5, bs):
            while offset>=len_X:
                offset -= len_X
            #take a batch
            batch_X, batch_y = X[offset:offset+bs], y[offset:offset+bs]
            #augment the dataset to increase and transform images
            batch_X, batch_y = data.augment_data(batch_X, batch_y)

            images = []
            s_angles = []
            for i in range(len(batch_X)):
                images.append(batch_X[i])
                s_angles.append(batch_y[i])

            X_batch = np.array(images)
            y_batch = np.array(s_angles)

            yield X_batch, y_batch
