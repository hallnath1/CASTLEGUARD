import pandas as pd
import numpy as np
from castle import CASTLE
from castle import Parameters
import csv_gen
import ml_utilities as mlu
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import app
# import keras
# from keras import backend
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout


sarray = []
ks = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
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

# def NN(X, Y):
# 	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# 	layers = [Dense(256, activation='relu', use_bias='true', input_dim=8),
# 			  Dense(64, activation='relu', use_bias='true', input_dim=256),
# 			  Dense(256, activation='relu', use_bias='true', input_dim=64),
# 			  Dense(1, activation='sigmoid', use_bias='true', input_dim=256),
# 			]
# 	model = Sequential(layers)
# 	# model.summary()
# 	model.compile(loss='binary_crossentropy', optimizer="adam" , metrics=['accuracy'])
# 	model_data = model.fit(X_train, Y_train, epochs=1000, batch_size=64, verbose=0)
# 	score = model.evaluate(X_test, Y_test, batch_size=64)

# 	print("Test Loss: {}".format(score[0]))
# 	print("Test Accuracy: {}".format(score[1]))

def handler(value: pd.Series):
    sarray.append(value.data)    

def main():
    args = app.parse_args()
    print("Loading in data")
    frame = pd.read_csv("adult.csv")
    cat = {
        "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked","?"],
        "maritalstatus": ['Married-civ-spouse', "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse","?"],
        "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces","?"],
        "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried","?"],
        "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black","?"],
        "sex": ["Male", "Female","?"],
        "nativecountry": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands","?"],
        "salary": [">50K", "<=50K"]
        }
    frame["pid"] = frame.index
    headers = ["age", "workclass", "fnlwgt", "maritalstatus", "educationnum", "occupation", "relationship", "race", "sex", "nativecountry", "capitalgain", "capitalloss", "hoursperweek"]
    extended_headers = ["spcage","minage","maxage", "spcworkclass","minworkclass","maxworkclass", "spcfnlwgt","minfnlwgt","maxfnlwgt", "spcmaritalstatus","minmaritalstatus","maxmaritalstatus", "spceducationnum","mineducationnum","maxeducationnum", "spcoccupation","minoccupation","maxoccupation", "spcrelationship","minrelationship","maxrelationship", "spcrace","minrace","maxrace", "spcsex","minsex","maxsex", "spcnativecountry","minnativecountry","maxnativecountry", "spccapitalgain","mincapitalgain","maxcapitalgain", "spccapitalloss","mincapitalloss","maxcapitalloss", "spchoursperweek","minhoursperweek","maxhoursperweek"]
    sensitive_attr = "salary"
    total = 0
    data = frame
    print("Processing Data")
    processed = mlu.process(data, cat)
    print("Processed Data")
    processed[sensitive_attr]=processed[sensitive_attr].astype('int')
    for i in ks:
        valid = validation(processed[headers], processed[sensitive_attr], i)
        print("K={} Accuracy: {}%".format(i, round(valid*100), 5))
        total += valid
    print("Average Accuracy for Pre-CASTLE: {}%".format(round((total/len(ks))*100, 5)))

    frame["pid"] = frame.index
    args.k = 1000
    args.l = 1
    args.delta = 10000
    args.mu = 100
    args.beta = 50
    Phi = [1, 10, 100, 1000]
    Big_Beta = [0.35, 0.5, 0.75, 1]
    acc_list = []
    print("Size: {}".format(frame.shape))
    print("Starting Loop")
    for args.phi in Phi:
        print("Phi: {}".format(args.phi))
        avg_acc_list = []
        for args.big_beta in Big_Beta:
            print("Big Beta: {}".format(args.big_beta))
            average = 0
            for loop in range(0, 10):
                frame = pd.read_csv("adult.csv")
                print("Processing Data")
                processed = mlu.process(frame, cat)
                print("Processed Data")
                processed[sensitive_attr]=processed[sensitive_attr].astype('int')
                processed["pid"] = processed.index
                global sarray
                sarray = []
                params = Parameters(args)
                stream = CASTLE(handler, headers, sensitive_attr, params)
                print("Starting CASTLE")
                counter = 0
                for(_, row) in processed.iterrows():
                    counter += 1
                    stream.insert(row)
                while(counter <= args.delta):
                    print("Cycling")
                    counter+=1
                    stream.cycle()
                print("Finished CASTLE")
                print(len(sarray))
                dataframes = []
                for s in sarray:
                    df = s.to_frame().transpose()
                    dataframes.append(df)
                avg = pd.concat(dataframes, ignore_index=True, sort=True)
                avg_features = avg[extended_headers]
                total = 0
                avg[sensitive_attr]=avg[sensitive_attr].astype('int')
                for i in ks:
                    valid = validation(avg_features, avg[sensitive_attr], i)
                    # print("K={} Accuracy: {}%".format(i, round(valid*100), 5))
                    total+=valid
                average += (total/9)
                print("Phi: {}, BBeta: {}, Average Accuracy: {}%".format(args.phi,args.big_beta, round((total/9)*100), 5))
            avg_acc_list.append(average/10)
            print("Overall Average: {}%".format((average/10)*100))
        acc_list.append(np.array(avg_acc_list))


    X, Y = np.meshgrid(Big_Beta, np.log(Phi))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.array(acc_list), rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_xlabel("Big Beta")
    ax.set_ylabel("Log(Phi)")
    ax.set_zlabel('Average Accuracy of KNN for Predicting Salary')
    plt.savefig("OrigData.png")
    # plt.show()

main()
