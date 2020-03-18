import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(42)
rn.seed(12345)
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from castle import CASTLE
from castle import Parameters
import ml_utilities as mlu
import app
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

print("Set threads")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

tf.random.set_seed(1234)

sarray = []

def NN(X_train, X_test, Y_train, Y_test, size=8):
    layers = [Dense(256, activation='relu', use_bias='true', input_dim=size),
              Dense(64, activation='relu', use_bias='true', input_dim=256),
              Dense(256, activation='relu', use_bias='true', input_dim=64),
              Dense(1, activation='sigmoid', use_bias='true', input_dim=256),
            ]
    model = Sequential(layers)
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam" , metrics=['accuracy'])
    model_data = model.fit(X_train, Y_train, epochs=1000, batch_size=64, verbose=0)
    score = model.evaluate(X_test, Y_test, batch_size=64)
    predictions = model.predict_classes(X_test)

    print("Test Loss: {}".format(score[0]))
    print("Test Accuracy: {}%".format(score[1]*100))

    results = confusion_matrix(Y_test, predictions)
    print ('Confusion Matrix :')
    print(results)
    print ('Classification Report : ')
    print (classification_report(Y_test, predictions))
    print('AUC-ROC:',roc_auc_score(Y_test, predictions))
    print('LOGLOSS Value is',log_loss(Y_test, predictions))

    return roc_auc_score(Y_test, predictions)

def handler(value: pd.Series):
    sarray.append(value.data) 

def main():
    args = app.parse_args()

    frame = pd.read_csv("diabetes.csv")
    headers=["pregnancies","glucose","bloodPressure","skinThickness","insulin","bmi","diabetesPedigree","age"]
    sensitive_attr = "outcome"
    X_train, X_test, Y_train, Y_test = train_test_split(frame[headers], frame[sensitive_attr], test_size=0.3)
    print("Normal Data")
    NN(X_train, X_test, Y_train, Y_test)
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
            train = X_train
            train[sensitive_attr] = Y_train
            train['pid'] = train.index
            global sarray
            sarray = []
            params = Parameters(args)
            stream = CASTLE(handler, headers, sensitive_attr, params)
            print("CASTLE START")
            counter=0
            for(_, row) in train.iterrows():
                counter+=1
                stream.insert(row)
            while(counter <= args.delta):
                counter+=1
                stream.cycle()
            print("CASTLE END")
            grped = mlu.average_group(sarray)
            acc = NN(grped[headers], X_test,grped[sensitive_attr], Y_test)
            avg_acc_list.append(acc)
        acc_list.append(np.array(avg_acc_list))
    print(acc_list)
    X, Y = np.meshgrid(Big_Beta, np.log(Phi))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.array(acc_list), rstride=1, cstride=1, cmap='winter', edgecolor='none')
    ax.set_xlabel("Big Beta")
    ax.set_ylabel("Log(Phi)")
    ax.set_zlabel('AUC-ROC')
    plt.show()
main()