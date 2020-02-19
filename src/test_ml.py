import pandas as pd
from castle import CASTLE
import csv_gen
import ml_utilities as mlu
from sklearn.neighbors import KNeighborsClassifier

def normalise(dataset, ax=0):
    return (dataset - dataset.mean(axis=ax))/dataset.std(axis=ax)

def handler(value: pd.Series):
	print("RECIEVED VALUE: {}".format(value))

if __name__=="__main__":

	
	# frame = pd.read_csv(args.filename).sample(20)
	
	# headers = list(frame.columns.values)
	
	# stream = CASTLE(handler, headers, args.k, args.delta, args.beta)
	
	# for(_, row) in frame.iterrows():
	# 	stream.insert(row)

	frame = pd.read_csv(csv_gen.generate_output_data("test")).sample(20)
	sarray = []
	for i in range(0, 20):
		row = frame.iloc[i]
		sarray.append(row)
	print("Average of series objects: \n{}".format(mlu.average_group(sarray)))


