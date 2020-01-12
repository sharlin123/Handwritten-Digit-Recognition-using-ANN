

import keras

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)

import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(figsize=(8,8))
for i in range(64):
  ax=fig.add_subplot(8,8,i+1)
  ax.imshow(x_train[i])
plt.show()

from keras.models import Sequential
from keras.layers import Dense

X_train=x_train.reshape(60000,784)

X_test=x_test.reshape(10000,784)

Y_train=keras.utils.to_categorical(y_train,10)
Y_test=keras.utils.to_categorical(y_test,10)

model=Sequential()
model.add(Dense(units=512,input_dim=784,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=1000,epochs=50)

predictions=model.predict(X_test)

model.evaluate(X_test,Y_test)

model.evaluate(X_train,Y_train)