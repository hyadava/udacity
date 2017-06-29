import sys, os
import tables

#load train and test data

h5file = tables.open_file('./trdata/drive_data.h5', mode='r', title="drive_data") 
X_train = h5file.get_node("/train_data").read()
y_train = h5file.get_node("/train_labels").read()
X_test = h5file.get_node("/test_data").read()
y_test = h5file.get_node("/test_labels").read()

print('X_train shape ', X_train.shape)
print('X_test shape ', X_test.shape)

h5file.close()

from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

input_shape = X_train.shape[1:]
height = input_shape[0]
width = input_shape[1]
channel = input_shape[2]

model = None

try:
	model = load_model('./trdata/model_snapshot.h5')
	print("loading existing model")

except:
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
				input_shape=(height, width, channel),
				output_shape=(height, width, channel)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	model.summary()
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_absolute_error'])

model.fit(X_train, y_train, nb_epoch=300, batch_size=64, validation_split=0.2, shuffle=True)
score = model.evaluate(X_test, y_test, batch_size=64)
print('score ', score[0], ' accuracy ', score[1])

model_file = './trdata/model.json'
model_weights = './trdata/model.h5'

model.save('./trdata/model_snapshot.h5')

import json

with open(model_file, "w") as mf:
	json.dump(model.to_json(), mf)
mf.close()

model.save_weights(model_weights)
