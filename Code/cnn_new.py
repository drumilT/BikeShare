from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.model_selection import train_test_split
# pip install -q tensorflow-gpu==2.0.0-beta1
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from error_function import travel_data_error, travel_data_accuracy

###Everything is prety similar to cnn.py please refer it
##Data input here is a 168*167 array such that it 167 points are the station points in the clusters

exp = False
input_dim_row = 168
input_dim_col = 167
label_dim = 13
time_inp = 1
epoch = None
denselayers = 5

file_out = open("../cnn_new.txt","w")



def get_data():

    fout = open("../usefulData/" + str(time_inp) + "_inp_data_out7am_red.bin", "rb")
    fin = open("../usefulData/" + str(time_inp) + "_inp_data_in_167_mem.bin", "rb")
    data_in = pickle.load(fin)
    data_out = pickle.load(fout)
    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, test_size=0.1, random_state=20)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    fout.close()
    fin.close()
    return (X_train, y_train), (X_test, y_test)


def abserr(y_true, y_pred):
    y_pred = np.array(y_pred)
    val = 100 * (np.mean(np.abs(y_pred - y_true) / np.abs(np.mean(y_true))))
    return val

def run():
    file_out.write(str(epoch)+'epoch '+str(denselayers)+"dense"+str(time_inp)+"time_inp")
    (train_images, train_labels), (test_images, test_labels) = get_data()

    train_images = train_images.reshape((train_images.shape[0], input_dim_row, input_dim_col, 1))
    test_images = test_images.reshape((test_images.shape[0],  input_dim_row, input_dim_col, 1))

    train_labels = train_labels.reshape((train_labels.shape[0], label_dim * label_dim))
    test_labels = test_labels.reshape((test_labels.shape[0], label_dim * label_dim))

    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(input_dim_row, input_dim_col, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(20, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(20, (3, 3), activation='relu'))

    model.summary()

    #model.add(layers.Flatten(input_shape=(time_inp*input_dim,input_dim,1)))
    model.add(layers.Flatten())

    model.add(layers.Dense( 13000, activation='relu'))
    model.add(layers.Dense( 5000, activation='relu'))
    model.add(layers.Dense( 2000, activation='relu'))
    model.add(layers.Dense( 500, activation='relu'))
    model.add(layers.Dense( 250, activation='relu'))
    model.add(layers.Dense(label_dim * label_dim, activation='relu'))

    model.summary()

    model.compile(optimizer='adam',
                  loss="mse",
                  metrics=['accuracy', 'mse'])

    # x = model.predict(test_images[0].reshape((1,19,19,1)))
    # print(x.shape)
    history = model.fit(train_images, train_labels, epochs=epoch)

    test_loss, test_acc, b = model.evaluate(test_images, test_labels)
    file_out.write(str(test_loss))
    file_out.write(str(test_acc))
    pred = model.predict(test_images)

    count = 0

    # pred = np.array(pred).reshape((len(pred), label_dim,label_dim))

    with open(
            "../usefulData/pred/" + str(time_inp) + "hr_inp" + str(denselayers) + "dense_layers" + str(epoch) + "epochsCNN_new",
            "wb") as f:
        pickle.dump(pred, f)

    arre = []
    arra = []

    plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch '+str(denselayers)+"dense"+str(time_inp)+"time_inpCNN167")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['mse'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch'+str(denselayers)+"dense"+str(time_inp)+"time_inpCNN167")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    for t, s in zip(pred, test_labels):
        print(count)

        arre.append(travel_data_error(s, t))
        arra.append(travel_data_accuracy(s, t))
        t = np.around(t)
        s = np.around(s)
        t = t.reshape((label_dim, label_dim))
        s = s.reshape((label_dim, label_dim))
        #plt.imshow(t)
        #plt.colorbar()
        #plt.savefig("../Graph/CNN_new/" + str(count) + '-Pred-graph.png')
        #plt.imshow(s)

        #plt.savefig("../Graph/CNN_new/" + str(count) + '-Actual-graph.png')
        #plt.close()
        count += 1
    print(arre)
    print(np.mean(np.array(arre)))
    file_out.write(str(np.mean(np.array(arre))))
    print(arra)
    print(np.mean(np.array(arra)))
    file_out.write(str(np.mean(np.array(arra))))
    file_out.write("\n")
epochs = [10,30,50,70,100,300,700,1000,1500,5000]
for t in epochs:
    epoch = t
    run()