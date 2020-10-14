# -*- coding: utf-8 -*-
"""
@author: udits
"""

#Fetch label attributes from word vectors and save it in numpy array

from gensim.models import KeyedVectors
import numpy as np

def fetch_wordvec():
    #importing pretrained google word2vec embedding
    filename = 'data/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    class_to_index = []
    with open('data/classes.txt') as f:
        index = 0
        for line in f:
            class_name = line.split('\t')[1].strip()
            class_to_index.append(class_name)


    vec_array = np.zeros((50,300))
    for animal,i in zip(class_to_index,range(0,50)):
        for char in animal:
            if char == '+':
                animal = animal.replace('+', '_')
                #Catching key error if any class is not present in pretrained embeddings
            try:         
                vec_array[i] = model[animal]
            except KeyError:
                print(f"{animal} not present in Google word2vec.")
                if animal == 'persian_cat' or animal == 'siamese_cat':
                    vec_array[i] = model['cat']
                if animal == 'blue_whale':
                    vec_array[i] = model['whale']
                    
                    
    np.save('vec_array.npy', vec_array)
                    
         