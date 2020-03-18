import pandas as pd
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
import os
import app

insert_time_list = []
counter = 0
filename = "test"
rows=1000
sarray = []

def normalise(dataset):
	return (dataset - dataset.mean())/dataset.std()

def accuracy(pred, actual):
#     Go through each row and inspect each value
#     for each of those values, compare them and increment correct counter
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

def handler(value: pd.Series):
	insert_time_list.append(counter)
	sarray.append(value.data)

def ml_test():
	print("test1")
	args = app.parse_args()
	global counter
	counter = 0
	frame = pd.read_csv(csv_gen.generate(filename,
										rows=rows,
										headers=["Age","GPA", "HoursPW", "EducationLvl", "Employed"],
										datatypes=["int120","float5", "int56", "int6", "int2"]))
	headers = ["Age","GPA", "HoursPW", "EducationLvl"]
	params = Parameters(args)
	sensitive_attr = "Employed"
	stream = CASTLE(handler, headers, sensitive_attr, params)

	for(_, row) in frame.iterrows():
		counter+=1
		stream.insert(row)

	# A function which tells us if there are any tuples which haven't been outputted yet

	avg = mlu.average_group(sarray)
	assert type(avg) is pd.DataFrame
	avg_features = avg[["Age", "HoursPW", "EducationLvl", "GPA"]]
	avg_norm = (avg_features - avg_features.mean()) / (avg_features.std())
	total = 0
	for i in range(1, 10):
		valid = validation(avg_norm, avg[sensitive_attr], i)
		total+=valid
	print("Average Accuracy for CASTLE: {}".format(total/9))
	os.remove("{}.csv".format(filename))


def test_process():
	frame = pd.read_csv(csv_gen.generate(filename,
									rows=1,
									headers=["Age","GPA", "HoursPW", "Education", "Employed"],
									datatypes=["int120","float5", "int56", "edu", "int2"],
									categorical={"edu":["PhD", "Masters", "Bachelors", "Secondary", "Primary"]}))
	cat = {"Education":["PhD", "Masters", "Bachelors", "Secondary", "Primary"]}
	keys = list(frame.keys())
	processed = mlu.process(frame, cat).iloc[0]
	frame_s = frame.iloc[0]
	for key in keys:
		if key in cat:
			assert cat[key].index(frame_s[key]) == processed[key]
		else:
			assert frame_s[key] == processed[key]
	os.remove("{}.csv".format(filename))

