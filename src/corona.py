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

	params = Parameters(args.k, args.delta, args.beta, args.mu, args.l)
	stream = CASTLE(handler, headers, sensitive_attr, params)
	
	for(_, row) in processed.iterrows():
		stream.insert(row)

	avg = mlu.average_group(sarray, [('CountryRegion', np.int64), ('pid', np.int64)])
	print(avg)

main()