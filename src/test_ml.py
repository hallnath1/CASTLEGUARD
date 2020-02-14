import pandas as pd
from castle import CASTLE
import csv_gen


def handler(value: pd.Series):
	print("RECIEVED VALUE: {}".format(value))

if __name__=="__main__":

	
	# frame = pd.read_csv(args.filename).sample(20)
	
	# headers = list(frame.columns.values)
	
	# stream = CASTLE(handler, headers, args.k, args.delta, args.beta)
	
	# for(_, row) in frame.iterrows():
	# 	stream.insert(row)

	frame = pd.read_csv(csv_gen.generate_output_data("test")).sample(20)
	print(frame)


