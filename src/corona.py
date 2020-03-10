import pandas as pd
import numpy as np
from castle import CASTLE
from castle import Parameters
import csv_gen
import ml_utilities as mlu
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import os
import app

sarray = []

def handler(value: pd.Series):
	sarray.append(value.data)

def normalise(dataset):
	return (dataset - dataset.mean())/dataset.std()

def main():
	args = app.parse_args()
	frame = pd.read_csv("covid_19_data.csv")[["ObservationDate","CountryRegion","LastUpdate","Confirmed", "Recovered", "Deaths"]]
	headers=["ObservationDate","CountryRegion","LastUpdate","Confirmed", "Recovered"]
	sensitive_attr = "Deaths"

	frame['ObservationDate']= pd.DatetimeIndex(frame['ObservationDate']).astype(np.int64)

	frame['LastUpdate']= pd.DatetimeIndex(frame['LastUpdate']).astype(np.int64)
	frame['pid'] = frame.index
	cat = {
		"CountryRegion": frame.CountryRegion.unique().tolist()
	}
	processed = mlu.process(frame, cat)
	print(processed)

	X = normalise(processed[headers])
	Y = processed[sensitive_attr]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	n_cols = X_train.shape[1]

	model = Sequential()
	model.add(Dense(256, activation="relu", input_shape=(n_cols,)))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer="Adam" , metrics=['accuracy'])
	model_data = model.fit(X_train, Y_train, epochs=10, batch_size=128)
	eval_model=model.evaluate(X_test, Y_test, batch_size=64)
	print("Pre-CASTLE Test Loss: {}".format(round(eval_model[0], 5)))
	print("Pre-CASTLE Test Accuracy: {}%".format(round(eval_model[1]*100, 5)))

	params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
	stream = CASTLE(handler, headers, sensitive_attr, params)
	
	for(_, row) in processed.iterrows():
		stream.insert(row)

	avg = mlu.average_group(sarray, [('CountryRegion', np.int64), ('pid', np.int64)])
	print(avg)

	X = normalise(avg[headers])
	Y = avg[sensitive_attr]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	n_cols = X_train.shape[1]

	model = Sequential()
	model.add(Dense(256, activation="relu", input_shape=(n_cols,)))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer="Adam" , metrics=['accuracy'])
	model_data = model.fit(X_train, Y_train, epochs=10, batch_size=128)
	eval_model=model.evaluate(X_test, Y_test, batch_size=64)
	print("Post-CASTLE Test Loss: {}".format(round(eval_model[0], 5)))
	print("Post-CASTLE Test Accuracy: {}%".format(round(eval_model[1]*100, 5)))

main()