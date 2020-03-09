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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import os
import app
from typing import List

sarray = []

def handler(value: pd.Series):
	sarray.append(value.data)

def normalise(dataset: pd.DataFrame) -> pd.DataFrame:
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
			

def main():
	args = app.parse_args()
	frame = pd.read_csv("fifa19.csv").head(200)[["Age", "Nationality", "Potential", "Club", "Value", "Wage", "Position", "Crossing", "Finishing", "HeadingAccuracy", "ShortPassing", "Volleys", "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance", "ShotPower", "Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions", "Positioning", "Vision", "Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling", "GKKicking", "GKPositioning", "GKReflexes", "Overall"]]
	headers=["Age", "Nationality", "Potential", "Club", "Value", "Wage", "Position", "Crossing", "Finishing", "HeadingAccuracy", "ShortPassing", "Volleys", "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance", "ShotPower", "Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions", "Positioning", "Vision", "Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling", "GKKicking", "GKPositioning", "GKReflexes"]
	sensitive_attr = "Overall"



	frame['pid'] = frame.index

	cat = {
		"Club": frame.Club.unique().tolist(),
		"Nationality": frame.Nationality.unique().tolist(),
		"Position": frame.Position.unique().tolist(),
	}
	processed = mlu.process(frame, cat)
	print(processed)
	from_short_num(processed, ["Wage", "Value"])

	X = processed[headers]
	Y = processed["Overall"]
	big = Y.max()
	low = Y.min()
	print("Big: {}, Small: {}".format(big, low))

	sc = StandardScaler()
	X = sc.fit_transform(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	hot_y_train = keras.utils.to_categorical(Y_train)
	model = Sequential()
	model.add(Dense(64))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(big+1, activation="softmax"))
	model.compile(loss='categorical_crossentropy', optimizer="Adam" , metrics=['accuracy'])
	model_data = model.fit(X_train, hot_y_train, epochs=10, batch_size=64)
	eval_model=model.evaluate(X_train, hot_y_train)
	print(eval_model)

	# params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
	# stream = CASTLE(handler, headers, sensitive_attr, params)
	
	# for(_, row) in processed.iterrows():
	# 	stream.insert(row)

	# avg = mlu.average_group(sarray,[("pid", np.int64), ("Overall", np.int64), ("Club", np.int64), ("Nationality", np.int64), ("Position", np.int64)])
	# print(avg)

main()