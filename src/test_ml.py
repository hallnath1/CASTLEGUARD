import pandas as pd
from castle import CASTLE
from castle import Parameters
import csv_gen
import ml_utilities as mlu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.optimizers import SGD
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

def ml_test1():
	print("test1")
	args = app.parse_args()
	global counter
	counter = 0
	frame = pd.read_csv(csv_gen.generate(filename,
										rows=rows,
										headers=["Age","GPA", "HoursPW", "EducationLvl", "Employed"],
										datatypes=["int120","float5", "int56", "int6", "int2"]))
	headers = ["Age","GPA", "HoursPW", "EducationLvl"]
	params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
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

def ml_test2():
	args = app.parse_args()
	global sarray
	global counter
	counter = 0
	sarray = []
	print("test2")
	frame = pd.read_csv("diabetes.csv")
	headers=["pregnancies","glucose","bloodPressure","skinThickness","insulin","bmi","diabetesPedigree","age"]
	sensitive_attr = "outcome"

	total = 0
	for i in range(1, 10):
		valid = validation(frame, frame[sensitive_attr], i)
		total += valid
	print("Average Accuracy for Regular Data: {}".format(total/9))
	
	frame["pid"] = frame.index
	params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
	stream = CASTLE(handler, headers, sensitive_attr, params)

	for(_, row) in frame.iterrows():
		counter+=1
		stream.insert(row)

	avg = mlu.average_group(sarray)
	avg_features = avg[headers]
	avg_norm = (avg_features - avg_features.mean()) / (avg_features.std())
	total = 0
	for i in range(1, 10):
		valid = validation(avg_norm, avg[sensitive_attr], i)
		total+=valid
	print("Average Accuracy for CASTLE: {}".format(total/9))

def ml_test3():
	args = app.parse_args()
	global sarray
	global counter
	counter = 0
	sarray = []
	print("test3")
	frame = pd.read_csv("diabetes.csv")
	headers=["pregnancies","glucose","bloodPressure","skinThickness","insulin","bmi","diabetesPedigree","age"]
	sensitive_attr = "outcome"
	layers = [Dense(256, activation='relu', use_bias='true', input_dim=8),
			  Dense(64, activation='relu', use_bias='true', input_dim=256),
			  Dense(256, activation='relu', use_bias='true', input_dim=64),
			  Dense(1, activation='sigmoid', use_bias='true', input_dim=256),
			]
	frame[headers] = normalise(frame[headers])
	train = frame[:int(len(frame)*0.75)]
	test = frame[int(len(frame)*0.75):]

	train_labels = train[sensitive_attr]
	train = train[headers]
	
	test_labels = test[sensitive_attr]
	test = test[headers]

	model = Sequential(layers)
	# model.summary()
	model.compile(loss='binary_crossentropy', optimizer="adam" , metrics=['accuracy'])
	model_data = model.fit(train, train_labels, epochs=1000, batch_size=64, verbose=0)

	score = model.evaluate(test, test_labels, batch_size=64)
	y_pred_classes = model.predict_classes(test)
	training_epoch = model_data.epoch
	training_loss = model_data.history['loss']
	try:
		training_accuracy = model_data.history['accuracy']
	except: 
		training_accuracy = model_data.history['acc']

	print("Pre-CASTLE Test Loss: {}".format(score[0]))
	print("Pre-CASTLE Test Accuracy: {}".format(score[1]))

	frame["pid"] = frame.index
	params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
	stream = CASTLE(handler, headers, sensitive_attr, params)

	for(_, row) in frame.iterrows():
		counter+=1
		stream.insert(row)

	avg = mlu.average_group(sarray)
	avg[headers] = normalise(avg[headers])
	train = avg[:int(len(frame)*0.75)]
	test = avg[int(len(frame)*0.75):]

	train_labels = train[sensitive_attr]
	train = train[headers]
	
	test_labels = test[sensitive_attr]
	test = test[headers]

	model = Sequential(layers)
	# model.summary()
	model.compile(loss='binary_crossentropy', optimizer="adam" , metrics=['accuracy'])
	model_data = model.fit(train, train_labels, epochs=1000, batch_size=64, verbose=0)

	score = model.evaluate(test, test_labels, batch_size=64)
	y_pred_classes = model.predict_classes(test)
	training_epoch = model_data.epoch
	training_loss = model_data.history['loss']
	try:
		training_accuracy = model_data.history['accuracy']
	except: 
		training_accuracy = model_data.history['acc']

	print("Post-CASTLE Test Loss: {}".format(score[0]))
	print("Post-CASTLE Test Accuracy: {}".format(score[1]))


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


ml_test1()
ml_test2()
ml_test3()
