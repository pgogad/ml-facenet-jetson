from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(1), W_regularizer=l2(0.01))
model.add(Activation('softmax'))
model.compile(loss='squared_hinge',
              optimizer='adadelta',
              metrics=['accuracy'])


# model = Sequential()
# model.add(Dense(512, activation='relu'))
# model.add(Dense(3), W_regularizer=l2(0.01))
# model.add(Activation('linear'))
#
# model.compile(loss='hinge', optimizer='adadelta', metrics=['accuracy'])
# model.build(input_shape=512)
# print(model.summary())
