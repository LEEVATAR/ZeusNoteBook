import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 1. 데이터 - ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,  
    validation_split=0.2
)

test_datagen = ImageDataGenerator(                  #평가만 해야하기 때문에 증폭할 필요성이 없다.
    rescale=1./255
)                                   

# flow_from_directory
tg_path = './Data/cat_dog/training_set/'
val_path = './Data/cat_dog/test_set/'
pic_path = './Data/'

train_generator = train_datagen.flow_from_directory(
    tg_path,
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    val_path,
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',  # rps 데이터는 'categorical'로 바꿔야함
    subset='validation',
    # validation_split=0.2   # ImageDataGenerater에서 정의 
)                                   

# 테스트할 이미지 구성
pic_test = train_datagen.flow_from_directory(
    pic_path,
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)

print(train_generator[0][0].shape)  # (5, 150, 150, 3)
print(validation_generator[0][0].shape) # (5, 150, 150, 3)
print(pic_test[0][0].shape)   # (5, 150, 150, 3)

# x = datasets.data 
# y = datasets.target
print(train_generator[0][0])    # x 데이터 =>x_train   
print(train_generator[0][1])    # y 데이터 = y_train 

x_train = train_generator[0][0]       
y_train = train_generator[0][1]
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout,Conv2D
from keras.layers import MaxPooling2D, Dropout

model = Sequential()
# Conv2D
# model.add(Conv2D(64, (3,3), input_shape=(150, 150, 3), 
#                  activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일,훈련
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
'''          
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=2,
    # validation_split=0.2, # fit_generator 안에는 validation_split이 없음
    validation_steps=10,
    validation_data=validation_generator
)
'''
hist = model.fit ( 
    x_train, y_train, epochs=2,
    validation_steps=10,
    # validation_split=0.2,
    validation_data=validation_generator   # model.fit에서 validation_data 사용가능
)
  
#4. 평가, 예측
accuracy = hist.history['accuracy']
loss = hist.history['loss']
val_accuracy = hist.history['val_accuracy']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])
print('val_loss : ', val_loss[-1])