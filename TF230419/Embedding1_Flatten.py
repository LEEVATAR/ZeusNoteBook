#2. 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(31, 10, input_length=5))  
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))