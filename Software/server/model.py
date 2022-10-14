from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
# MLP for Pima Indians Dataset with grid search via sklearn
# import keras
# from keras import models
# from keras import layers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import base 
import numpy
 


def generate_models():
    models = [ generate_RF_model(),  generate_SVM_model(), ]
    models_name = ["RF", "SVM"]
    return models, models_name

def generate_dl_models(class_num):
    ms = [ generate_cnn_model(class_num)]
    models_name = ["cnn"]
    return ms, models_name

def generate_lstm_model(class_num, optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def generate_cnn_model(class_num, optimizer='rmsprop', init='glorot_uniform'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(class_num))
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    return model

def generate_RF_model():
    return RandomForestClassifier(n_estimators=100, max_depth = 30)

def dl_clone(model):
	m = models.clone_model(model)
	m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
	return m

def generate_SVM_model():
    return SVC(gamma='auto', kernel='rbf')


if __name__ == '__main__':
    print(generate_models())
    