import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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

train_generator = train_datagen.flow_from_directory(
    '디렉토리 경로',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '디렉토리 경로',
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    subset='validation'
)                                   

print(train_generator[0][0].shape)  # 719
print(validation_generator[0][0].shape) # 308

# x = datasets.data 
# y = datasets.target
# print(xy_train[0][0])        
# print(xy_train[0][1])          
# print(xy_train[0][2])          


#2. 모델
from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# 컴파일,훈련
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
             
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=200,
    validation_steps=10,
)