from os import write
import joblib
import numpy as np
from numpy import loadtxt
from model import generate_models
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dsp_utils import DSPUtils
from statistics import mean, stdev
import pandas as pd
import json
import matplotlib.pyplot as plt


DATA_COLLECTION_FOLDER = "./demo_data/"
THRESHOLD = 3
SAMPLE_RATE = 1024
BUFFER_SIZE = 512
SHIFT_SIZE = 128 
# participants = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
participants = ["p0"]
# items = ["table", "drawer", 'cuttingboard'] ### what items are the model trained for?
items = ['cuttingboard'] ### what items are the model trained for?
### mapping from items to their corresponding activities
item_to_activities = {"table":["Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" ,  "Dispensing Tape"],
                    "drawer":["Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping", "Dispensing Tape"],
                    "cuttingboard":["Chopping","Stirring"]
                     } 

def printOverallAccuracy(cm):
    acc = []
    for i in range(len(cm)):
        if sum(cm[i]) > 0:
            acc.append(cm[i][i]/ sum(cm[i]))
    print('acc')
    print(mean(acc))
    print("std")
    print(stdev(acc))

def train(): 
    all_data = []
    for item in items:
        for participant in participants:
            for a in item_to_activities[item]:
                file_name =DATA_COLLECTION_FOLDER+item + "/" + participant + "/"+ a +'.json'
                with open(file_name, 'r+') as file:
                    print(file)
                    data = json.load(file)
                    for d in data[:]:
                        new_data = {}
                        new_data["participant"] = participant
                        new_data["item"] = item
                        new_data["activity"] = a
                        record_sig = np.array(d["record_data"])
                        signal, fft_windows = DSPUtils.segment_along_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        new_data["feature"] =  DSPUtils.extract_feature(signal, fft_windows)

                        all_data.append(new_data)


    df = pd.DataFrame(all_data)
    models, model_names = generate_models()
    saved_models = []
    i = 0
    
    for m in models:
        for item in items:
            item_data = df.loc[(df['item'] == item)]
            scaler = MinMaxScaler()
            features = scaler.fit_transform(item_data["feature"].to_list())
           
            strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
            model = clone(m)
            y_pred = cross_val_predict(model, features, item_data["activity"].to_list(), cv=strat_k_fold)
            cm = confusion_matrix(item_data["activity"].to_list(), y_pred, labels=item_to_activities[item])
            print(item)
            print(model_names[i])
            print("Two Fold Acc:")
            print(cm)
            printOverallAccuracy(cm)

            model = clone(m)
            model.fit(features, item_data["activity"].to_list())
            model_file_name = model_names[i] + '_model'
            joblib.dump(model, './model/'+item+ '_' + model_file_name)
        i+=1

if __name__ == '__main__':
    train()
    
