import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ImageDataGenerator
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
tg_path = './Data/horse-or-human/'
# val_path = './Data/horse-or-human/'  # 필요없음

xy_generator = train_datagen.flow_from_directory(
    tg_path,
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    shuffle=True,
    subset='training'
    # Found 822 images belonging to 2 classes.
)

validation_generator = test_datagen.flow_from_directory(
    tg_path,
    target_size=(150,150),  #사이즈조절가능
    batch_size=5,
    class_mode='binary',
    # subset='validation'
)                                   

print(xy_generator[0][0].shape)  # x 데이터
print(xy_generator[0][1].shape)  # y 데이터 

x_train = xy_generator[0][0]
y_train = xy_generator[0][1]   
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )

#2. 모델
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout,Conv2D
from keras.layers import MaxPooling2D, Dropout

model = Sequential()
# Conv2D
model.add(Conv2D(64, (3,3), input_shape=(150, 150, 3), 
                 activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.25))
model.add(Flatten())  #LSTM으로 대체 가능 
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# 컴파일,훈련
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
             
hist = model.fit_generator(
    xy_generator,
    steps_per_epoch=20,
    epochs=2,
    validation_steps=10,
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
