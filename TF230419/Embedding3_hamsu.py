#2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Input

model = Sequential()
input1 = Input(shape=(5,))
dense1 = Embedding(input_dim=30, output_dim=10, input_length=5)(input1)
dense2 = LSTM(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()