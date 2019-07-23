from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
#pip install -q tensorflow-gpu==2.0.0-beta1
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
tf.compat.v1.enable_eager_execution()
from error_function import travel_data_accuracy, travel_data_error
input_dim = 173
label_dim = 13
time_inp = 1
epoch = 70
denselayers = 11

def get_data():
    fout = open("../usefulData/"+str(time_inp)+"_inp_data_out729_red.bin" , "rb")
    fin = open("../usefulData/"+str(time_inp)+"_inp_data_in729_red.bin" , "rb")
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

(train_images, train_labels), (test_images, test_labels) = get_data()
train_images = train_images.reshape((train_images.shape[0], input_dim, 1))
test_images = test_images.reshape((test_images.shape[0], input_dim, 1))
train_labels = train_labels.reshape((train_labels.shape[0], label_dim*label_dim))
test_labels = test_labels.reshape((test_labels.shape[0], label_dim*label_dim))


model = models.Sequential()
model.add(layers.Flatten(input_shape=(173,1)))
jump = int((input_dim- label_dim*label_dim) / denselayers)
for i in range(1,denselayers):
   model.add(layers.Dense( input_dim - jump* i, activation='relu'))
model.add(layers.Dense(label_dim*label_dim, activation='relu'))

model.summary()
model.compile(optimizer='adam',
              loss= 'mse' ,
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epoch)
test_loss, test_acc = model.evaluate(test_images, test_labels)
pred = model.predict(test_images)
count = 0

with open("../usefulData/pred/"+str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochsANN" , "wb") as f:
    pickle.dump(pred,f)
arre =[]
arra =[]
for t,s in zip(pred, test_labels):
    print(count)
    arre.append(travel_data_error(s,t))
    arra.append(travel_data_accuracy(s,t))
    t = np.around(t).reshape((label_dim,label_dim))
    s = s.reshape((label_dim,label_dim))
    plt.imshow(t)
    plt.colorbar()
    plt.savefig("../Graph/ANN/" + str(count) + '-Pred-graph.png')
    plt.imshow(s)

    plt.savefig("../Graph/ANN/" + str(count) + '-Actual-graph.png')
    count += 1
    plt.close()


print(arre)
print(np.mean(np.array(arre)))
print(arra)
print(np.mean(np.array(arra)))
