# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:52:09 2018

@author: wangchenghan
"""

import my_dict
from skimage import transform
from skimage import io

from PIL import Image
#import glob
import os
import pickle
import random
import numpy as np
import sys
import os
print(sys.path)
train_dict = my_dict.get_train_dict()
predict_dict = my_dict.get_predict_dict()

train_dir = '../AgriculturalDisease_trainingset'
validate_dir = '../AgriculturalDisease_validationset'

def get_imagelist(data_dir, image_list=[]):
    '''
    获取文件夹下图片列表
    '''
    items = os.listdir(data_dir)
    for i in items:
        name = os.path.join(data_dir,i)
        if os.path.splitext(i)[-1] == '.jpg' or os.path.splitext(i)[-1] == '.JPG' or os.path.splitext(i)[-1] == '.png' or os.path.splitext(i)[-1] == '.PNG':
            image_list.append(name)
        elif os.path.isdir(name):
            image_list = get_imagelist(name,image_list)
    return image_list

def isdisease(dirname):
    '''
    根据给定的文件名（包括地址）判断其疾病类别
    '''
    names = dirname.split('\\')
    for i in range(len(names)-1):
        di_name = ''
        di_name = di_name.join(names[-2-i:-1])
        di_name = di_name.replace(' ','')
        if di_name  in train_dict:
#            print(di_name)  
            return train_dict[di_name]
    print('NOOOOOOOOO!')
    return None
    

def get_dataset_dict(image_list):
    '''
    根据图片列表中图片所属的文件夹制作标注，并按原image_list的顺序返回标注列表
    '''
    data_dict = {}
    for i in image_list:
        data_dict[i] = isdisease(i)
    return data_dict
#        print(isdisease(i)) 

def get_dataset():
    AgriculturalDisease_dict = {}
    if not os.path.exists('AgriculturalDisease.pickle'):
        validate_imagelist = get_imagelist(validate_dir, [])
        random.shuffle(validate_imagelist)
        validate_dataset = get_dataset_dict(validate_imagelist)

        train_imagelist = get_imagelist(train_dir, [])
        random.shuffle(train_imagelist)
        train_dataset = get_dataset_dict(train_imagelist)

        AgriculturalDisease_dict['train'] = train_dataset
        AgriculturalDisease_dict['validate'] = validate_dataset
        with open('AgriculturalDisease.pickle','wb') as file:
            pickle.dump(AgriculturalDisease_dict, file)
    else:
        with open('AgriculturalDisease.pickle','rb') as file:
            AgriculturalDisease_dict = pickle.load(file)   
    return AgriculturalDisease_dict
    
def load_data(img_rows=224, img_cols=224, train_samples=3000, validate_samples=100):
    '''
    读取农业疾病数据集   
    '''
    AgriculturalDisease_dict = get_dataset()
    X_train = np.zeros((train_samples, img_rows, img_cols, 3), dtype=np.uint8)
    Y_train = np.zeros((train_samples, 51), dtype=np.uint8)
    
    X_valid = np.zeros((validate_samples, img_rows, img_cols, 3), dtype=np.uint8)
    Y_valid = np.zeros((validate_samples, 51), dtype=np.uint8)
    
    print(Y_train.shape)
    print(Y_valid.shape)

#######################################################   
    dataset = AgriculturalDisease_dict['train']
    count = 0
    for i in dataset:
        im =io.imread(i)
        #io.imshow(im)
       # plt.show()
        im = transform.resize(im, (img_rows, img_cols),preserve_range=True)
        X_train[count,:,:,:] = im
#        print(dataset[i])
        Y_train[count,dataset[i]] = 1
#        print(Y_train[count])
        count = count + 1
        print('train_image:',count)
        if count>train_samples-1:
            break
    dataset = AgriculturalDisease_dict['validate']
    count = 0
    for i in dataset:
        im =io.imread(i)
        im = transform.resize(im, (img_rows, img_cols),preserve_range=True)
        X_valid[count,:,:,:] = im
        Y_valid[count,dataset[i]] = 1
#        print(Y_valid[count])
        count = count + 1
        print('validate_image:',count)
        if count>validate_samples-1:
            break
    return (X_train,Y_train),(X_valid,Y_valid)
#######################################################
#######################################################
#    dataset = AgriculturalDisease_dict['train']
#    for i in dataset:
#        im = Image.open(i)
#        im = im.resize((img_rows, img_cols))
#        X_train.append(im)
#        Y_train.append(dataset[i])
#    dataset = AgriculturalDisease_dict['validate']
#    for i in dataset:
#        im =Image.open(i)
#        im = im.resize((img_rows, img_cols))
#        X_valid.append(im)
#        Y_valid.append(dataset[i])
#    return X_train,Y_train,X_valid,Y_valid
########################################################
########################################################
#    dataset = AgriculturalDisease_dict['train']
#    for i in dataset:
#        X_train.append(i)
#        Y_train.append(dataset[i])
#    dataset = AgriculturalDisease_dict['validate']
#    for i in dataset:
#        X_valid.append(i)
#        Y_valid.append(dataset[i])
#    return X_train,Y_train,X_valid,Y_valid
##########################################################

def main():
    #image_list = get_datalist(train_dir)
    ##print(len(image_list))
    #data_dict = get_label(image_list)
    #print(data_dict)
    X_train,Y_train,X_valid,Y_valid = load_data(img_rows=224, img_cols=224, train_samples=10, validate_samples=2)
    print(X_train.shape, Y_train.shape, X_valid.shape,Y_valid.shape)
#    io.imshow(X_train[0,:,:,:])
#    io.imshow(X_valid[0,:,:,:])
#    print(X_train)

if __name__ == '__main__':
    main()