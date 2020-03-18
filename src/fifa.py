import pandas as pd
import numpy as np
from castle import CASTLE
from castle import Parameters
import csv_gen
import ml_utilities as mlu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
try:
	import keras
	from keras import backend
	from keras.models import Sequential
	from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
except:
	print("Can't import")
from sklearn.model_selection import train_test_split
import re
import os
import app
from typing import List

sarray = []

def handler(value: pd.Series):
	sarray.append(value.data)

def normalise(dataset: pd.DataFrame):
	return (dataset - dataset.mean())/dataset.std()

def from_short_num(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame :
	values = []
	for(_, row) in data.iterrows():
		for col in cols:
			m = re.search("[\d]+M|[\d]+[.][\d]+M", row[col])
			k = re.search("[\d]+K", row[col])
			if m:
				row[col] = float(m.group(0).strip("M"))*1000000
			elif k:
				row[col] = float(k.group(0).strip("K"))*1000
			else:
				row[col] = float(row[col])

def NN(X, Y):
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
	return eval_model[1]

def accuracy(pred, actual):
	true=0
	false=0
	actual_val = actual.tolist()
	pred_val = pred.tolist()
	for i, val in enumerate(pred_val):
		if val == actual_val[i]:
			true+=1
		else:
			false+=1

	return true/(true+false)

def validation(features, labels, nb):
	neigh = KNeighborsClassifier(n_neighbors=nb)
	kf = KFold(n_splits=10)
	kf.get_n_splits(features)
	pred_val = []
	for train_index, test_index in kf.split(features):
		X_train, X_test = features.iloc[train_index], features.iloc[test_index]
		y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
		neigh.fit(X_train, y_train)
		pred = neigh.predict(X_test)
		pred_val.append(accuracy(pred, y_train))
	return sum(pred_val) / len(pred_val)

			

def main():
	args = app.parse_args()
	frame = pd.read_csv("fifa19.csv")[["Age", "Nationality","Wage", "Value","Potential","Club","Position","Overall"]]
	headers=["Age", "Nationality","Potential","Wage", "Value","Club","Position"]
	sensitive_attr = "Overall"

	frame['pid'] = frame.index

	cat = {
		"Club": frame.Club.unique().tolist(),
		"Nationality": frame.Nationality.unique().tolist(),
		"Position": frame.Position.unique().tolist(),
	}
	processed = mlu.process(frame, cat)
	from_short_num(processed, ["Wage", "Value"])

	X = normalise(processed[headers])
	Y = processed[sensitive_attr]
	Y=Y.astype('int')

	print("Pre-CASTLE Test Accuracy: {}%".format(round(NN(X,Y)*100, 5)))

	print("Pre-CASTLE KNN")

	total = 0
	for i in range(1, 10):
		valid = validation(X,Y, i)
		print("K={} Accuracy: {}%".format(i, round(valid*100), 5))
		total+=valid
	print("Average Accuracy for Pre-CASTLE: {}%".format(round((total/9)*100, 5)))

	params = Parameters(args)
	stream = CASTLE(handler, headers, sensitive_attr, params)
	for(_, row) in processed.iterrows():
		stream.insert(row)
	avg = mlu.average_group(sarray,[("pid", np.int64), ("Overall", np.int64), ("Club", np.int64), ("Nationality", np.int64), ("Position", np.int64)])

	X = normalise(avg[headers])
	Y = avg[sensitive_attr]
	Y=Y.astype('int')

	print("Post-CASTLE Test Accuracy: {}%".format(round(NN(X,Y)*100, 5)))

	print("Post-CASTLE KNN")

	total = 0
	for i in range(1, 10):
		valid = validation(X,Y, i)
		print("K={} Accuracy: {}%".format(i, round(valid*100), 5))
		total+=valid
	print("Average Accuracy for Post-CASTLE: {}%".format(round((total/9)*100, 5)))

main()