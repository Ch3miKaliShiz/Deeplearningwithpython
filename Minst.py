# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:53:36 2022

@author: rkatw
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

x = np.array([np.linspace(0,100,100), np.linspace(0, 100, 100)])


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
         for j in range(x.shape[1]):
             x[i,j] - max(x[i,j], 0)
    return x

def naive_add(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] += y[i,j]
    return x

def naive_sub(x,y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] -= y[i,j]
    return x

