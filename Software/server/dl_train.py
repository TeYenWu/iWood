from os import write
import joblib
import numpy as np
from numpy import loadtxt
from dsp import convert_to_fft_windows
from sklearn import preprocessing
from model import generate_dl_models
from model import dl_clone
# from dsp import compute_relevant_features
from statistics import mean
import pandas as pd
import json
import matplotlib.pyplot as plt


BATCH_SIZE = 32
THRESHOLD = 3
SAMPLE_RATE = 1024
BUFFER_SIZE = 512
SHIFT_SIZE = 128 
participants = ["zheer", "yun", "wu", "baying", "chixiang", "quijia"]
userinput_list = ["Tap", "Swipe", "Knock", "Slap"]
# activity_list = userinput_list
# thing_to_activity = {"table":userinput_list,
#                     "drawer":userinput_list,
#                     # "cuttingboard":["Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling"]
#                      }
activity_list = ["Tap", "Swipe", "Knock", "Slap", "Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" , "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Dispensing Tape", "Grating"]

thing_to_activity = {"table":["Tap", "Swipe", "Knock", "Slap", "Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping" , "Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Dispensing Tape", "Grating"],
                    "drawer":["Tap", "Swipe", "Knock", "Slap", "Writing", "Erasing","Staple", "Pen Sharpening",  "Pumping", "Dispensing Tape"],
                    "cuttingboard":["Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling", "Grating"]
                     }
things = ["table", "drawer", "cuttingboard"]

def printOverallAccuracy(cm):
    acc = []
    for i in range(len(cm)):
        if sum(cm[i]) > 0:
            acc.append(cm[i][i]/ sum(cm[i]))
    print(mean(acc))

def saveConfusionMatrix(cm, name):
    fig = plt.figure()
    plt.matshow(cm)
    plt.title(name)
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('./cm/confusion_matrix_'+str(name)+'.jpg')
    plt.clf()

def train(): 
    all_data = []
    for thing in things:
        for participant in participants:

            for index, a in enumerate(thing_to_activity[thing]):
                file_name ='./activity_data/'+thing + "/" + participant + "/"+ a +'.json'
                with open(file_name, 'r+') as file:
                    print(file)
                    data = json.load(file)
                    for d in data:
                        new_data = {}
                        new_data["participant"] = participant
                        new_data["thing"] = thing
                        new_data["activity"] = index
                        # record_sig = down_sample(np.array(d["record_data"]))
                        record_sig = np.array(d["record_data"])
                        # record_sig = record_sig*1024
                        # record_sig = record_sig.astype(int)
                        # record_sig = record_sig.astype(float)
                        # record_sig = record_sig/1024
                        fft_windows = convert_to_fft_windows(record_sig, d["background"], BUFFER_SIZE, SHIFT_SIZE)
                        new_data["feature"] =  fft_windows

                        all_data.append(new_data)


    df = pd.DataFrame(all_data)

    models,model_names = generate_dl_models(len(activity_list))
    saved_models = []
    i = 0
    
    for m in models:
        cross_user_on_everything_done = False
        cross_thing = {"Actual": [], "Predict": []}
        for thing in things:
            print(thing)
            cross_user_on_thing = {"Actual": [], "Predict": []}
            cross_user_on_everything = {"Actual": [], "Predict": []}
            for participant in participants:
                ### cross-user accuracy on specific thing
                other_user = df.loc[(df['participant'] != participant) & (df['thing'] == thing)]
                target_user = df.loc[(df['participant'] == participant) & (df['thing'] == thing)]
                model = dl_clone(m)
                train_x = other_user["feature"].to_numpy()
                train_x.astype(np.float32)
                model.fit(, other_user["activity"].to_numpy().astype(np.float32), batch_size=BATCH_SIZE, epochs=10, 
                    validation_split=0.5)
                probs = model.predict(target_user['feature'].to_numpy().astype(np.float32))
                y = np.argmax(probs, axis=-1) 
                print(participant)
                cm = confusion_matrix(target_user['activity'].to_list(), list(y), labels=thing_to_activity[thing])
                print(cm)
                printOverallAccuracy(cm)
                saveConfusionMatrix(cm, participant + '_' + thing)


                ### cross-user on everything 
                if not cross_user_on_everything_done:
                    # print("cross_user_on_everything")
                    model = dl_clone(m)
                    model = dl_clone(m)
                    model.fit(other_user["feature"].to_numpy().astype(np.float32), other_user["activity"].to_numpy().astype(np.float32), batch_size=BATCH_SIZE, epochs=10, 
                        validation_split=0.5)
                    probs = model.predict(target_user['feature'].to_numpy().astype(np.float32))
                    y = np.argmax(probs, axis=-1) 
                    cross_user_on_everything["Actual"] += target_user['activity'].to_list()
                    cross_user_on_everything["Predict"] += list(y)


            ### plot confusion matrix for cross-user accuracies on specific thing
            cm = confusion_matrix(cross_user_on_thing["Actual"], cross_user_on_thing["Predict"], labels=thing_to_activity[thing])
            ### print cross-user accuracy on specific thing
            print(thing + " cross_user acc:")
            printOverallAccuracy(cm)
            print(cm)

            if not cross_user_on_everything_done:

                ### plot confusion matrix for cross-user accuracies
                cm = confusion_matrix(cross_user_on_everything["Actual"], cross_user_on_everything["Predict"], labels=activity_list)
                ### print cross-user accuracies on everything 
                print("cross_user_on_everything acc:")
                printOverallAccuracy(cm)
                print(cm)
                cross_user_on_everything_done = True
            

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
                model = dl_clone(m)
                model.fit(other_user["feature"].to_numpy().astype(np.float32), other_user["activity"].to_numpy().astype(np.float32), batch_size=BATCH_SIZE, epochs=10, 
                    validation_split=0.5)
                probs = model.predict(target_user['feature'].to_numpy().astype(np.float32))
                y = np.argmax(probs, axis=-1) 
                # print(y)
                cross_thing["Actual"] += other_thing['activity'].to_list()
                cross_thing["Predict"] += y

                ### plot confusion matrix for cross-thing accuracies
                cm = confusion_matrix(cross_thing["Actual"], cross_thing["Predict"], labels=thing_to_activity[thing])
                print(thing)
                print(model_names[i])
                print(cm)
                ### print cross-thing accuracies on everything 
                print("crossthing acc:")
                printOverallAccuracy(cm)

            thing_data = df.loc[(df['thing'] == thing)]

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
    
