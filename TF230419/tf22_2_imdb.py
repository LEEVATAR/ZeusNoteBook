from keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000                     # num_words는 단어의 빈도수 설정, embedding의 input_dim에 넣어주면 됨
)

print(x_train)
print(x_train.shape, x_test.shape)    #(25000,) (25000,) => 25000개씩의 리스트  
print(y_train)    
print(np.unique(y_train, return_counts=True))    
print(y_train.shape, y_test.shape)    #(25000,) (25000,) => 25000개씩의 리스트 
print(len(np.unique(y_train)))        # 2개의 label이 있음, 다중분류 

print(type(x_train), type(y_train))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))               # <class 'list'>
# print(x_train[0].shape)             # AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0]))                # 218
print(len(x_train[1]))                # 189

# print(len(max(x_train)))
print('리뷰의 최대길이 : ', max(len(i) for i in x_train))   # 리뷰의 최대길이 :  2494
print('리뷰의 평균길이 : ', sum(map(len, x_train)) / len(x_train))  # 리뷰의 평균길이 :  238.71364

#전처리
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                        #shape=(25000,) => (25000, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)                        
y_test = to_categorical(y_test)                        

print(x_train.shape, y_train.shape)     # (25000, 100) (25000, 2)
print(x_test.shape, y_test.shape)       # (25000, 100) (25000, 2)