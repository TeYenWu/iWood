from os import write
import joblib
import numpy as np
from numpy import loadtxt
from model import generate_models
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from micromlgen import port

from sklearn.preprocessing import normalize
from dsp import extract_feature_from_raw_signal, segment, extract_feature, down_sample
# from dsp import compute_relevant_features
from statistics import mean, stdev
import pandas as pd
import json
import matplotlib.pyplot as plt


THRESHOLD = 3
SAMPLE_RATE = 1024
BUFFER_SIZE = 512
SHIFT_SIZE = 128 
participants = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
# userinput_list = ["Tap", "Swipe", "Knock", "Slap"]
# activity_list = userinput_list
# thing_to_activity = {"table":userinput_list,
#                     "drawer":userinput_list
#                      }
# things = ["table", "drawer"]
activity_list = ["Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" ,  "Dispensing Tape", "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Grating"]

thing_to_activity = {"table":["Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" ,  "Dispensing Tape", "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Grating"],
                    "drawer":["Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping", "Dispensing Tape"],
                    "cuttingboard":["Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Grating"]
                     }
things = ["table", "drawer", "cuttingboard"]

def printOverallAccuracy(cm):
    acc = []
    for i in range(len(cm)):
        if sum(cm[i]) > 0:
            acc.append(cm[i][i]/ sum(cm[i]))
    print('acc')
    print(mean(acc))
    print("std")
    print(stdev(acc))

def plot_and_save_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    if len(target_names) > 10:
        plt.rcParams.update({'font.size': 40})
    else:
        plt.rcParams.update({'font.size': 50})

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20), dpi = 120)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()

    # if target_names is not None:
    #     tick_marks = np.arange(len(target_names))
    #     plt.xticks(tick_marks, target_names, rotation=45)
    #     plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm * 100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                if cm[i, j] == 100:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./cm/confusion_matrix_'+str(title)+'.jpg')
    plt.clf()
    # plt.show()

def train(): 
    all_data = []
    for thing in things:
        for participant in participants:
            for a in thing_to_activity[thing]:
                file_name ='./activity_data/'+thing + "/" + participant + "/"+ a +'.json'
                with open(file_name, 'r+') as file:
                    print(file)
                    data = json.load(file)
                    for d in data:
                        new_data = {}
                        new_data["participant"] = participant
                        new_data["thing"] = thing
                        new_data["activity"] = a
                        # record_sig = down_sample(np.array(d["record_data"]))
                        record_sig = np.array(d["record_data"])
                        # record_sig = record_sig*1024
                        # record_sig = record_sig.astype(int)
                        # record_sig = record_sig.astype(float)
                        # record_sig = record_sig/1024
                        signal, fft_windows = segment(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        # print(fft_windows.shape)
                        new_data["feature"] =  extract_feature(signal, fft_windows)

                        all_data.append(new_data)


    df = pd.DataFrame(all_data)
    models, model_names = generate_models()
    saved_models = []
    i = 0
    
    for m in models:
        cross_user_on_everything_done = False
        within_user_on_everything_done = False
        cross_thing = {"Actual": [], "Predict": []}
        for thing in things:
            print(thing)
            cross_user_on_thing = {"Actual": [], "Predict": []}
            cross_user_on_everything = {"Actual": [], "Predict": []}
            within_user_on_thing = {"Actual": [], "Predict": []}
            within_user_on_everything = {"Actual": [], "Predict": []}
            for participant in participants:



                ### cross-user accuracy on specific thing
                other_user = df.loc[(df['participant'] != participant) & (df['thing'] == thing)]
                target_user = df.loc[(df['participant'] == participant) & (df['thing'] == thing)]
                model = clone(m)
                model.fit(other_user["feature"].to_list(), other_user["activity"].to_list())
                y = model.predict(target_user['feature'].to_list())
                cm = confusion_matrix(target_user['activity'].to_list(), list(y), labels=thing_to_activity[thing])
                cross_user_on_thing["Actual"] += target_user['activity'].to_list()
                cross_user_on_thing["Predict"] += list(y)
                # print(participant)
                # print("Cross user:")
                # print(cm)
                # printOverallAccuracy(cm)

                # saveConfusionMatrix(cm, participant + '_' + thing)

                ### within-user accuracy on thing 
                model = clone(m)
                strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
                y_pred = cross_val_predict(model, target_user["feature"].to_list(), target_user["activity"].to_list(), cv=strat_k_fold)
                cm = confusion_matrix(target_user["activity"].to_list(), y_pred, labels=thing_to_activity[thing])
                within_user_on_thing["Actual"] += target_user['activity'].to_list()
                within_user_on_thing["Predict"] += list(y_pred)
                # print(participant)
                # print("Two Fold Acc:")
                # print(cm)
                # printOverallAccuracy(cm)



                ### cross-user accuracy on everything 
                if not cross_user_on_everything_done:
                    # print("cross_user_on_everything")
                    other_user = df.loc[(df['participant'] != participant)]
                    target_user = df.loc[(df['participant'] == participant)]
                    model = clone(m)
                    model.fit(other_user["feature"].to_list(), other_user["activity"].to_list())
                    y = model.predict(target_user['feature'].to_list())
                    cross_user_on_everything["Actual"] += target_user['activity'].to_list()
                    cross_user_on_everything["Predict"] += list(y)
 
                ### within-user accuracy on everything
                if not within_user_on_everything_done:
                    # print("cross_user_on_everything")
                    target_user = df.loc[(df['participant'] == participant)]
                    model = clone(m)
                    strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
                    y_pred = cross_val_predict(model, target_user["feature"].to_list(), target_user["activity"].to_list(), cv=strat_k_fold)
                    cm = confusion_matrix(target_user["activity"].to_list(), y_pred, labels=thing_to_activity[thing])
                    within_user_on_everything["Actual"] += target_user['activity'].to_list()
                    within_user_on_everything["Predict"] += list(y_pred)

            # ### plot confusion matrix for cross-user accuracies on specific thing
            # cm = confusion_matrix(cross_user_on_thing["Actual"], cross_user_on_thing["Predict"], labels=thing_to_activity[thing])
            # plot_and_save_confusion_matrix(cm, thing_to_activity[thing], thing + "-cross-user")
            # ### print cross-user accuracy on specific thing
            # print(thing + " cross_user acc:")
            # printOverallAccuracy(cm)
            # print(cm)

            #  ### plot confusion matrix for within-user accuracies on specific thing
            # cm = confusion_matrix(within_user_on_thing["Actual"], within_user_on_thing["Predict"], labels=thing_to_activity[thing])
            # ### print cross-user accuracy on specific thing
            # plot_and_save_confusion_matrix(cm, thing_to_activity[thing], thing + "-within-user")
            # print(thing + " within_user acc:")
            # printOverallAccuracy(cm)
            # print(cm)

            if not cross_user_on_everything_done:

                ### plot confusion matrix for cross-user accuracies
                cm = confusion_matrix(cross_user_on_everything["Actual"], cross_user_on_everything["Predict"], labels=activity_list)
                ### print cross-user accuracies on everything 
                print("cross_user_on_everything acc:")
                plot_and_save_confusion_matrix(cm, activity_list, "cross_user_on_everything")
                printOverallAccuracy(cm)
                print(cm)
                cross_user_on_everything_done = True

            if not within_user_on_everything_done:
                ### plot confusion matrix for cross-user accuracies
                cm = confusion_matrix(within_user_on_everything["Actual"], within_user_on_everything["Predict"], labels=activity_list)
                ### print cross-user accuracies on everything 
                print("within_user_on_everything acc:")
                plot_and_save_confusion_matrix(cm, activity_list, "within_user_on_everything")
                printOverallAccuracy(cm)
                print(cm)
                within_user_on_everything_done = True
            

            if thing != "table":
                ### cross-thing accuracy
                other_thing = df.loc[(df['thing'] != thing)  & (df['activity'].isin(thing_to_activity[thing]))]
                target_thing = df.loc[(df['thing'] == thing) & (df['activity'].isin(thing_to_activity[thing]))]
                # strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
                # acc = cross_val_score(model, features, labels, cv=strat_k_fold, scoring='recall_macro')
                # m = model.fit(other_thing["feature"].to_list(), other_thing["activity"].to_list())
                # y = list(model.predict(target_thing['feature'].to_list()))
                # # print(y)
                # cross_thing["Actual"] += target_thing['activity'].to_list()
                # cross_thing["Predict"] += y
                model = clone(m)
                model.fit(other_thing["feature"].to_list(), other_thing["activity"].to_list())
                y = list(model.predict(target_thing['feature'].to_list()))
                # print(y)
                cross_thing["Actual"] += target_thing['activity'].to_list()
                cross_thing["Predict"] += y

               

            thing_data = df.loc[(df['thing'] == thing)]
            
            # strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
            # model = clone(m)
            # y_pred = cross_val_predict(model, thing_data["feature"].to_list(), thing_data["activity"].to_list(), cv=strat_k_fold)
            # cm = confusion_matrix(thing_data["activity"].to_list(), y_pred, labels=thing_to_activity[thing])
            # print(thing)
            # print(model_names[i])
            # print("Two Fold Acc:")
            # print(cm)
            # printOverallAccuracy(cm)

         ### plot confusion matrix for cross-thing accuracies
        cm = confusion_matrix(cross_thing["Actual"], cross_thing["Predict"], labels=activity_list)
        plot_and_save_confusion_matrix(cm, activity_list, "cross_thing")
        print(model_names[i])
        print("crossthing acc:")
        print(cm)
        printOverallAccuracy(cm)
        ## print cross-thing accuracies on everything 
        

        # model = clone(m)
        # strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        # y_pred = cross_val_predict(model, df["feature"].to_list(), df["activity"].to_list(), cv=strat_k_fold)
        # cm = confusion_matrix(df["activity"].to_list(), y_pred, labels=activity_list)

        # print(model_names[i])
        # print("overall model")
        # print(cm)
        # printOverallAccuracy(cm)
        

        # model = clone(m)
        # strat_k_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        # y_pred = cross_val_predict(model, df["feature"].to_list(), df["thing"].to_list(), cv=strat_k_fold)
        # cm = confusion_matrix(df["thing"].to_list(), y_pred, labels=things)
        # print("thing detection")
        # print(cm)
        # printOverallAccuracy(cm)
        i += 1

    # for model_name, model in saved_models:
    #     model_file_name =  model_name + '_model'
    #     joblib.dump(model, './model/'+model_file_name)
    #     text = port(model)
    #     with open('./model_script/'+model_file_name+'.c','w') as model_script:
    #         model_script.write(text)
    #         model_script.close()

if __name__ == '__main__':
    train()
    # matrix = [
    # [83.5, 0, 0, 6.5, 0, 9, 0, 0.5, 0, 0, 0, 0.5],
    # [0, 79, 1.5, 2.5, 0.5, 4, 0, 0, 0, 2.5, 3.5, 6.5],
    # [0, 0, 94.5, 0, 3, 1.5, 0, 0, 0, 0, 1, 0],
    # [5, 2.5, 0, 83, 0.5, 5, 0, 0.5, 0, 0, 0.5, 3],
    # [3, 1.5, 4, 2.5, 72.5, 16, 0, 1, 0, 0, 0, 0.5],
    # [7, 1, 1, 2, 5.5, 83, 0, 0, 0, 0, 0, 0.5],
    # [0, 0.5, 0, 0, 0, 0, 97.5, 1.5, 0, 0, 0.5, 0],
    # [1, 0, 0.5, 0.5, 0, 0.5, 4.5, 93, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1, 0, 97.5, 0.5, 1, 0],
    # [0, 5.5, 0, 0, 0, 0, 0, 0, 2.5, 81.5, 9, 1.5],
    # [0.5, 2.5, 1, 0, 0, 0.5, 2.5, 0.5, 4.0, 7, 73.6, 8],
    # [1, 5, 0, 4, 0, 1, 0, 0, 0.5, 0, 2.5, 86],
    # ]
    # matrix = np.array(matrix)
    # printOverallAccuracy(matrix)
    # plot_and_save_confusion_matrix(matrix, activity_list, "within-user_mixed-item")

    
