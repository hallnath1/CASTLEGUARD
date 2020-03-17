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
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout


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

def NN(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    layers = [Dense(256, activation='relu', use_bias='true', input_dim=8),
              Dense(64, activation='relu', use_bias='true', input_dim=256),
              Dense(256, activation='relu', use_bias='true', input_dim=64),
              Dense(1, activation='sigmoid', use_bias='true', input_dim=256),
            ]
    model = Sequential(layers)
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam" , metrics=['accuracy'])
    model_data = model.fit(X_train, Y_train, epochs=1000, batch_size=64, verbose=0)
    score = model.evaluate(X_test, Y_test, batch_size=64)

    print("Test Loss: {}".format(score[0]))
    print("Test Accuracy: {}".format(score[1]))

def handler(value: pd.Series):
    sarray.append(value.data)    

def main():
    args = app.parse_args()

    frame = pd.read_csv("diabetes.csv")
    headers=["pregnancies","glucose","bloodPressure","skinThickness","insulin","bmi","diabetesPedigree","age"]
    extended_headers=["spcpregnancies","minpregnancies","maxpregnancies","spcglucose","minglucose","maxglucose","spcbloodPressure","minbloodPressure","maxbloodPressure","spcskinThickness","minskinThickness","maxskinThickness","spcinsulin","mininsulin","maxinsulin","spcbmi","minbmi","maxbmi","spcdiabetesPedigree","mindiabetesPedigree","maxdiabetesPedigree","spcage","minage","maxage"]
    sensitive_attr = "outcome"
    total = 0
    for i in ks:
        valid = validation(frame, frame[sensitive_attr], i)
        print("K={} Accuracy: {}%".format(i, round(valid*100), 5))
        total += valid
    print("Average Accuracy for Pre-CASTLE: {}%".format(round((total/9)*100, 5)))

    frame["pid"] = frame.index
    args.k = 7
    args.l = 1
    args.delta = 100
    args.mu = 100
    args.beta = 25
    Phi = [1, 10, 100, 1000]
    Big_Beta = [0.25, 0.5, 0.75, 1]
    acc_list = []
    for args.phi in Phi:
        print("Phi: {}".format(args.phi))
        avg_acc_list = []
        for args.big_beta in Big_Beta:
            print("Big Beta: {}".format(args.big_beta))
            average = 0
            for looping in range(0, 10):
                frame = pd.read_csv("diabetes.csv")
                frame["pid"] = frame.index
                
                global sarray
                sarray = []
                params = Parameters(args)
                stream = CASTLE(handler, headers, sensitive_attr, params)

                for(_, row) in frame.iterrows():
                    stream.insert(row)

                dataframes = []
                for s in sarray:
                    df = s.to_frame().transpose()
                    dataframes.append(df)
                avg = pd.concat(dataframes, ignore_index=True, sort=True)
                avg_features = avg[extended_headers]
                total = 0
                for i in ks:
                    valid = validation(avg_features, avg[sensitive_attr], i)
                    # print("K={} Accuracy: {}%".format(i, round(valid*100), 5))
                    total+=valid
                print("Accuracy: {}%".format((total/9)*100))
                average+=(total/9)
            print("Average Accuracy: {}%".format((average/10)*100))
            avg_acc_list.append(average/10)
        acc_list.append(np.array(avg_acc_list))

    X, Y = np.meshgrid(Big_Beta, np.log(Phi))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.array(acc_list), rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_xlabel("Big Beta")
    ax.set_ylabel("Log(Phi)")
    ax.set_zlabel('Average KNN Accuracy')
    plt.show()

main()