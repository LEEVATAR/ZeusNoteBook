from sre_parse import Tokenizer
from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(   # 로이터 기사 뉴스
    num_words=10000, test_split=0.2
)

print(x_train)
print(x_train.shape, x_test.shape)    #(8982,) (2246, ) => 8982개와 2246개의 리스트  => 총 11,228개
print(y_train)    
print(np.unique(y_train, return_counts=True))    
print(y_train.shape, y_test.shape)    #(8982,) (2246, ) => 8982개와 2246개의 리스트  => 총 11,228개
print(len(np.unique(y_train)))   # 46개의 label이 있음, 다중분류 

print(type(x_train), type(y_train))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))               # <class 'list'>
# print(x_train[0].shape)             # AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0]))                # 87
print(len(x_train[1]))                # 56


#전처리
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                        #shape=(8982,) => (8982, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)                        
y_test = to_categorical(y_test)                        

print(x_train.shape, y_train.shape)     # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)       # (2246, 100) (2246, 46)


#2. 모델 구성
# [실습] 시작!!!