#require sklearn, tensorflow and keras

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
#pip install -q tensorflow-gpu==2.0.0-beta1
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from error_function import travel_data_error,travel_data_accuracy

##Main Params
exp = False ## True if the data stored is an exponentiated form of the original OD graph
input_dim = 17 ## Expects a input_dim*input_dim array nas input
label_dim = 13 ## Expects a label_dim*label_dim array nas output
time_inp = 3 ## defines the hours of prior data you want the CNN to learn on
epoch = 100 ## defines to number of epochs for the CNN for training
denselayers = 11 ## Number of dense layers after convolution to be introduced linearly

file_out = open("../cnn.txt","a") ##file to store your results post run

def get_data():

    # Function to fetch data for the CNN , requires it the required files in the place as defined in open command
    # defined below
    # not much to change here if you are using hourly_data_fast to generate data
    if exp:
        src = "EXP"
    else:
        src = ""
    fout = open("../usefulData/"+str(time_inp)+"_inp_data_out7am"+src+"_red_mem.bin" , "rb")
    fin = open("../usefulData/"+str(time_inp)+"_inp_data_in7am"+src+"_red_mem.bin" , "rb")
    data_in = pickle.load(fin)
    data_out = pickle.load(fout)
    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, test_size=0.1, random_state=20)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    fout.close()
    fin.close()
    return (X_train,y_train), (X_test, y_test)

def abserr (y_true, y_pred):
    #Calculates a absolute relative error in the matrices
    #not used anywhere but jut defined in case of requirement
    y_pred = np.array(y_pred)
    val = 100* (np.mean(np.abs(y_pred - y_true) / np.abs(np.mean(y_true))))
    return val

def plot_test_pred(pred,test):
    #Plots the predicted and actual graphs
    count = 0
    for t,s in zip(pred, test):
        if exp :
            s = np.log(s)
            t = np.log(t)
        t = np.around(t)
        s = np.around(s)
        t = t.reshape((label_dim,label_dim))
        s = s.reshape((label_dim,label_dim))
        plt.imshow(t)
        plt.colorbar()
        plt.savefig("../Graph/CNN/" + str(count) + '-Pred-graph.png')
        plt.imshow(s)

        plt.savefig("../Graph/CNN/" + str(count) + '-Actual-graph.png')
        plt.close()

def calc_new_err_and_acc(pred , test_labels):
    #Uses the accuracy and neww error function in the error function file to calculate acc on the predicted labels
    arre =[]
    arra =[]

    for t,s in zip(pred, test_labels):
        if exp :
            s = np.log(s)
            t = np.log(t)

        arre.append(travel_data_error(s,t))
        arra.append(travel_data_accuracy(s,t))
    print(arre)
    print(np.mean(np.array(arre)))
    file_out.write(str(np.mean(np.array(arre))))
    print(arra)
    print(np.mean(np.array(arra)))
    file_out.write(str(np.mean(np.array(arra))))
    file_out.write("\n")

def run():
    print(str(time_inp) + "hr_inp" + str(denselayers) + "dense_layers" + str(epoch) + "epochsCNN")

    file_out.write(str(epoch) + 'epoch ' + str(denselayers) + "dense" + str(time_inp) + "time_inp")

    (train_images, train_labels), (test_images, test_labels) = get_data()

    train_images = train_images.reshape((train_images.shape[0], time_inp*input_dim, input_dim, 1))
    test_images = test_images.reshape((test_images.shape[0], time_inp*input_dim, input_dim, 1))

    train_labels = train_labels.reshape((train_labels.shape[0], label_dim*label_dim))
    test_labels = test_labels.reshape((test_labels.shape[0], label_dim*label_dim))


    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(time_inp*input_dim, input_dim, 1)))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))

    model.summary()

    model.add(layers.Flatten(input_shape=(time_inp*input_dim,input_dim,1)))
    model.add(layers.Flatten())

    jump = int((time_inp*input_dim*input_dim - label_dim*label_dim) / denselayers)

    for i in range(1,denselayers):
       model.add(layers.Dense( time_inp*input_dim*input_dim - jump* i, activation='relu'))

    model.add(layers.Dense(label_dim*label_dim, activation='relu'))

    model.summary()

    model.compile(optimizer='adam',
                  loss= "mse",
                  metrics= ['accuracy','mse'])

    history = model.fit(train_images, train_labels, epochs=epoch)

    test_loss, test_acc, b = model.evaluate(test_images, test_labels)
    file_out.write(str(test_loss))
    file_out.write(str(test_acc))
    pred = model.predict(test_images)

    #pred = np.array(pred).reshape((len(pred), label_dim,label_dim))

    with open("../usefulData/pred/"+str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochsCNN" , "wb") as f:
        pickle.dump(pred,f)

    #print(history.history.keys())

    plot_acc_vs_epoch(history)
    plot_mse_vs_epoch(history)
    calc_new_err_and_acc(pred,test_labels)
    plot_test_pred(pred,test_labels)
    del model

def plot_mse_vs_epoch(history):
    #funcion name defines it all
    plt.plot(history.history['mse'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch'+str(denselayers)+"dense"+str(time_inp)+"time_inpCNN")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_acc_vs_epoch(history):
    #function anme defines it
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch '+str(denselayers)+"dense"+str(time_inp)+"time_inpCNN")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def tuning():
    time_inps = [1,3,5]
    epochs = [50,100,500,1000]
    denselayerss =[5,7,9,11,13,15,17,20]

    global epoch,time_inp,denselayers
    for t in time_inps:
        for e in epochs:
            for d in denselayerss:
                if e >= d*5:
                    time_inp = t
                    epoch = e
                    denselayers = d
                    run()

def refined_tuning():
    finely_tuned = [(1, 7, 100), (1, 9, 100), (1, 17, 500), (3, 5, 50), (3, 9, 100), (3, 11, 100), (3, 13, 100), (3, 15, 100), (3, 17, 500), (5, 11, 100), (5, 13, 100)]

    global epoch, time_inp, denselayers
    for time,dense,epo in finely_tuned:
        jump = int(0.1 * epo)
        epochs = []
        for i in range(1,4):
            t = epo - i*jump
            s = epo + i*jump
            epochs.append(t)
            epochs.append(s)
        print(epochs)
        for e in epochs:
            time_inp = time
            epoch = e
            denselayers = dense
            run()

run()