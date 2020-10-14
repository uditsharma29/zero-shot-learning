# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:19:10 2020

@author: udits
"""
import numpy as np
#from gensim.models import KeyedVectors
import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
import cv2

class word2vec1(data.dataset.Dataset):
  def __init__(self, classes_file, transform):
    #importing pretrained google word2vec embedding
    # filename = 'data/GoogleNews-vectors-negative300.bin'
    # model = KeyedVectors.load_word2vec_format(filename, binary=True)

    # class_to_index = []
    # with open('data/classes.txt') as f:
    #     index = 0
    #     for line in f:
    #         class_name = line.split('\t')[1].strip()
    #         class_to_index.append(class_name)


    # vec_array = np.zeros((50,300))
    # for animal,i in zip(class_to_index,range(0,50)):
    #     for char in animal:
    #         if char == '+':
    #             animal = animal.replace('+', '_')
    #             #Catching key error if any class is not present in pretrained embeddings
    #         try:         
    #             vec_array[i] = model[animal]
    #         except KeyError:
    #             print(f"{animal} not present in Google word2vec.")
    #             if animal == 'persian_cat' or animal == 'siamese_cat':
    #                 vec_array[i] = model['cat']
    #             if animal == 'blue_whale':
    #                 vec_array[i] = model['whale']
    predicate_binary_mat = np.load('data/vec_array.npy')
    self.predicate_binary_mat = predicate_binary_mat
    self.transform = transform

    class_to_index = dict()
    # Build dictionary of indices to classes
    with open('data/classes.txt') as f:
      index = 0
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    img_names = []
    img_index = []
    with open('data/{}'.format(classes_file)) as f:
      for line in f:
        class_name = line.strip()
        FOLDER_DIR = os.path.join('data/JPEGImages', class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob(file_descriptor)

        class_index = class_to_index[class_name]
        count = 0
        for file_name in files:
          #if count < 400:
            img_names.append(file_name)
            img_index.append(class_index)
            count += 1
    self.img_names = img_names
    self.img_index = img_index

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)
    if im.shape != (3,224,224):
      print(self.img_names[index])

    im_index = self.img_index[index]
    im_predicate = self.predicate_binary_mat[im_index,:]
    return im, im_predicate, self.img_names[index], im_index

  def __len__(self):
    return len(self.img_names)

if __name__ == '__main__':
    dataset = word2vec1('testclasses.txt')