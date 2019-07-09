from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
#pip install -q tensorflow-gpu==2.0.0-beta1
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()

input_dim = 17
label_dim = 13
time_inp = 5
epoch = 500
denselayers = time_inp

def get_data():
    fout = open("../usefulData/"+str(time_inp)+"_inp_data_out7am_red.bin" , "rb")
    fin = open("../usefulData/"+str(time_inp)+"_inp_data_in7am_red.bin" , "rb")
    data_in = pickle.load(fin)
    data_out = pickle.load(fout)
    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, test_size=0.1, random_state=200)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    fout.close()
    fin.close()
    return (X_train,y_train), (X_test, y_test)

def abserr (y_true, y_pred):
    y_pred = np.array(y_pred)
    val = 100* (np.mean(np.abs(y_pred - y_true) / np.abs(np.mean(y_true))))
    return val

(train_images, train_labels), (test_images, test_labels) = get_data()
#tf.eagerly()
train_images = train_images.reshape((train_images.shape[0], time_inp*input_dim, input_dim, 1))
test_images = test_images.reshape((test_images.shape[0], time_inp*input_dim, input_dim, 1))

train_labels = train_labels.reshape((train_labels.shape[0], label_dim*label_dim))
test_labels = test_labels.reshape((test_labels.shape[0], label_dim*label_dim))

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(10, (2, 2), activation='relu', input_shape=(time_inp*input_dim, input_dim, 1)))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(30, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(30, (3, 3), activation='relu'))

#model.summary()

model.add(layers.Flatten())
jump = int((time_inp*input_dim*input_dim - label_dim*label_dim) / denselayers)

for i in range(1,denselayers):
    model.add(layers.Dense( time_inp*input_dim*input_dim - jump* i, activation='relu'))

model.add(layers.Dense(label_dim*label_dim, activation='relu'))

model.summary()

model.compile(optimizer='adam',
              loss= 'mse' ,
              metrics=['accuracy'])

#x = model.predict(test_images[0].reshape((1,19,19,1)))
#print(x.shape)
model.fit(train_images, train_labels, epochs=epoch)

test_loss, test_acc = model.evaluate(test_images, test_labels)

pred = model.predict(test_images)

count = 0

with open("../usefulData/pred/"+str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochs" , "wb") as f:
    pickle.dump(pred,f)

for t,s in zip(pred, test_labels):
    t = np.around(t).reshape((label_dim,label_dim))
    #print(t)
    #print(s)

    s = s.reshape((label_dim,label_dim))
    plt.imshow(t)
    if count == 0:
        plt.colorbar()
    plt.savefig("../Graph/" + str(count) + '-Pred-graph.png')
    plt.imshow(s)

    plt.savefig("../Graph/" + str(count) + '-Actual-graph.png')
    count += 1