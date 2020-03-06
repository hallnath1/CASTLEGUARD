import pandas as pd
from castle import CASTLE
import csv_gen
import ml_utilities as mlu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import os

insert_time_list = []
counter = 0
filename = "test"
rows=1000
sarray = []

def normalise(dataset, ax=0):
	return (dataset - dataset.mean(axis=ax))/dataset.std(axis=ax)

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
	print("RECIEVED VALUE: {}".format(value))
	insert_time_list.append(counter)
	sarray.append(value)

if __name__=="__main__":

	
	# frame = pd.read_csv(args.filename)
	
	# headers = list(frame.columns.values)
	
	# stream = CASTLE(handler, headers, args.k, args.delta, args.beta)
	
	# for(_, row) in frame.iterrows():
	#   counter+=1
	# 	stream.insert(row)

	# A function which tells us if there are any tuples which haven't been outputted yet

	frame = pd.read_csv(csv_gen.generate_output_data(filename,
													rows=rows, 
													headers=["Age","GPA", "HoursPW", "EducationLvl", "Employed"], 
													datatypes=["int120","float5", "int56", "int6", "int2"], 
													generalise=["Age","GPA", "HoursPW", "EducationLvl"]))
	for i in range(0, rows):
		row = frame.iloc[i]
		sarray.append(row)

	avg = mlu.average_group(sarray)
	avg_features = avg[["Age", "HoursPW", "EducationLvl", "GPA"]]
	avg_norm = (avg_features - avg_features.mean()) / (avg_features.std())
	for i in range(1, 10):
		print("Avg for k = "+str(i)+": "+str(validation(avg_norm, avg["Employed"], i)))
	os.remove("{}.csv".format(filename))