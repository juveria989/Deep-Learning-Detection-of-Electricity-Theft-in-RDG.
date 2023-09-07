from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from keras.layers import  MaxPooling2D
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Bidirectional,GRU

main = tkinter.Tk()
main.title("Deep Learning Detection of Electricity Theft Cyber-attacks in Renewable Distributed Generation") 
main.geometry("1000x650")

global filename
global dnn_model
global X, Y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier
global cnn_model

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global X, Y
    global le
    global dataset
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    dataset['client_id'] = pd.Series(le.fit_transform(dataset['client_id'].astype(str)))
    dataset['label'] = dataset['label'].astype('uint8')
    print(dataset.info())
    dataset.drop(['creation_date'], axis = 1,inplace=True)
    text.insert(END,str(dataset.head())+"\n\n")
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype('uint8')
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = Y.astype('uint8')
    text.insert(END,"Total records found in dataset to train Deep Learning : "+str(X.shape[0])+"\n\n")


def rocGraph(testY, predict, algorithm):
    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict,pos_label=1)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    plt.title(algorithm+" ROC Graph")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.show()

def runGRU():
    global X, Y
    Y1 = to_categorical(Y)
    Y1 = Y1.astype('uint8')
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)

    if os.path.exists('model/gru_model.json'):
        with open('model/gru_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            gru_model = model_from_json(loaded_model_json)
        json_file.close()
        gru_model.load_weights("model/gru_model_weights.h5")
        gru_model._make_predict_function()
    else:
        counts = np.bincount(Y1[:, 0])
        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]
        class_weight = {0: weight_for_0, 1: weight_for_1}
        gru_model = Sequential() #defining deep learning sequential object
        #adding GRU layer with 32 filters to filter given input X train data to select relevant features
        gru_model.add(Bidirectional(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
        #adding dropout layer to remove irrelevant features
        gru_model.add(Dropout(0.2))
        #adding another layer
        gru_model.add(Bidirectional(GRU(32)))
        gru_model.add(Dropout(0.2))
        #defining output layer for prediction
        gru_model.add(Dense(y_train.shape[1], activation='softmax'))
        #compile GRU model
        gru_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #start training model on train data and perform validation on test data
        hist = gru_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test),class_weight=class_weight)
        #save model weight for future used
        gru_model.save_weights('model/gru_model_weights.h5')
        model_json = gru_model.to_json()
        with open("model/gru_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    y_test = np.argmax(y_test, axis=1)
    predict = gru_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"GRU Precision : "+str(p)+"\n")
    text.insert(END,"GRU Recall    : "+str(r)+"\n")
    text.insert(END,"GRU FMeasure  : "+str(f)+"\n")
    text.insert(END,"GRU Accuracy  : "+str(f)+"\n\n")
    rocGraph(y_test, predict, "GRU")

def runCNN():
    global X, Y
    Y1 = to_categorical(Y)
    Y1 = Y1.astype('uint8')
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)
    global cnn_model
    if os.path.exists('model/cnn_model.json'):
        with open('model/cnn_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_model = model_from_json(loaded_model_json)
        json_file.close()
        cnn_model.load_weights("model/cnn_model_weights.h5")
        cnn_model._make_predict_function()          
    else:
        counts = np.bincount(Y1[:, 0])
        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]
        class_weight = {0: weight_for_0, 1: weight_for_1}
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_model.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(output_dim = 256, activation = 'relu'))
        cnn_model.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn_model.fit(X_train, y_train, batch_size=64, epochs=20, shuffle=True, verbose=2, validation_data=(X_test, y_test),class_weight=class_weight)
        cnn_model.save_weights('model/cnn_model_weights.h5')            
        model_json = cnn_model.to_json()
        with open("model/cnn_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
    y_test = np.argmax(y_test, axis=1)
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"CNN Precision : "+str(p)+"\n")
    text.insert(END,"CNN Recall    : "+str(r)+"\n")
    text.insert(END,"CNN FMeasure  : "+str(f)+"\n")
    text.insert(END,"CNN Accuracy  : "+str(f)+"\n\n")
    rocGraph(y_test, predict, "CNN")
   
def runDNN():
    text.delete('1.0', END)
    global X, Y
    global dnn_model
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    Y1 = to_categorical(Y)
    Y1 = Y1.astype('uint8')
    X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2, random_state=0)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            dnn_model = model_from_json(loaded_model_json)
        json_file.close()    
        dnn_model.load_weights("model/model_weights.h5")
        dnn_model._make_predict_function()   
        print(dnn_model.summary())        
    else:
        counts = np.bincount(Y1[:, 0])
        weight_for_0 = 1.0 / counts[0]
        weight_for_1 = 1.0 / counts[1]
        class_weight = {0: weight_for_0, 1: weight_for_1}
        dnn_model = Sequential() #creating RNN model object
        dnn_model.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform")) #defining one layer with 256 filters to filter dataset
        dnn_model.add(Dense(128, activation='relu', kernel_initializer = "uniform"))#defining another layer to filter dataset with 128 layers
        dnn_model.add(Dense(y_train.shape[1], activation='softmax',kernel_initializer = "uniform")) #after building model need to predict two classes such as normal or Dyslipidemia disease
        dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #while filtering and training dataset need to display accuracy 
        print(dnn_model.summary()) #display rnn details
        hist = cnn_model.fit(X_train, y_train, epochs=20, batch_size=64,class_weight=class_weight)
        dnn_model.save_weights('model/model_weights.h5')            
        model_json = dnn_model.to_json()
        with open("model/model.json", "w") as json_file:
          json_file.write(model_json)
        json_file.close()    
    y_test = np.argmax(y_test, axis=1)
    predict = dnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"DNN Precision : "+str(p)+"\n")
    text.insert(END,"DNN Recall    : "+str(r)+"\n")
    text.insert(END,"DNN FMeasure  : "+str(f)+"\n")
    text.insert(END,"DNN Accuracy  : "+str(f)+"\n\n")
    rocGraph(y_test, predict, "DNN")

def runCNNRandomForest():
    global cnn_model
    global classifier
    global X, Y
    global cnn_model
    print(cnn_model.summary())
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    extract = Model(cnn_model.inputs, cnn_model.layers[-2].output)
    XX = extract.predict(X1)
    print(XX.shape)
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2, random_state=0)
    X_train, X_test1, y_train, y_test1 = train_test_split(X_test, y_test, test_size=0.2, random_state=0)
    rfc = RandomForestClassifier(n_estimators=50)
    rfc.fit(X_test, y_test)
    classifier = rfc
    predict = rfc.predict(X_test1)
    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    a = accuracy_score(y_test1,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Extension CNN with Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Extension CNN with Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Extension CNN with Random Forest FMeasure  : "+str(f)+"\n")
    text.insert(END,"Extension CNN with Random Forest Accuracy  : "+str(f)+"\n\n")
    rocGraph(y_test1, predict, "Extension CNN + Random Forest")


def predict():
    global classifier
    global cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    test = test.values
    data = test
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    extract = Model(cnn_model.inputs, cnn_model.layers[-2].output)
    test = extract.predict(test)
    predict = classifier.predict(test)
    for i in range(len(predict)):
        if predict[i] == 1:
            text.insert(END,str(data[i])+" ===> record detected as ELECTRICITY THEFT\n\n")
        if predict[i] == 0:
            text.insert(END,str(data[i])+" ===> record NOT detected as ELECTRICITY THEFT\n\n")     
    
def graph():
    df = pd.DataFrame([['DNN','Precision',precision[0]],['DNN','Recall',recall[0]],['DNN','F1 Score',fscore[0]],['DNN','Accuracy',accuracy[0]],
                       ['GRU','Precision',precision[1]],['GRU','Recall',recall[1]],['GRU','F1 Score',fscore[1]],['GRU','Accuracy',accuracy[1]],
                       ['CNN','Precision',precision[2]],['CNN','Recall',recall[2]],['CNN','F1 Score',fscore[2]],['CNN','Accuracy',accuracy[2]],
                       ['Extension CNN+RF','Precision',precision[3]],['Extension CNN+RF','Recall',recall[3]],['Extension CNN+RF','F1 Score',fscore[3]],['Extension CNN+RF','Accuracy',accuracy[3]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Deep Learning Detection of Electricity Theft Cyber-attacks in Renewable Distributed Generation', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Electricity Theft Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500,y=100)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Run Feed Forward Neural Network", command=runDNN)
cnnButton.place(x=200,y=150)
cnnButton.config(font=font1) 

cnnrfButton = Button(main, text="Run RNN GRU Algorithm", command=runGRU)
cnnrfButton.place(x=500,y=150)
cnnrfButton.config(font=font1)

cnnsvmButton = Button(main, text="Run Deep Learning CNN Algorithm", command=runCNN)
cnnsvmButton.place(x=200,y=200)
cnnsvmButton.config(font=font1)

rfButton = Button(main, text="Run Extension CNN + Random Forest", command=runCNNRandomForest)
rfButton.place(x=500,y=200)
rfButton.config(font=font1)

predictButton = Button(main, text="Predict Electricity Theft", command=predict)
predictButton.place(x=200,y=250)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=500,y=250)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=800,y=250)
exitButton.config(font=font1)

                            

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
