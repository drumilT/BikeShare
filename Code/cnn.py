from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
#pip install -q tensorflow-gpu==2.0.0-beta1
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def get_data():
    fout = open("../usefulData/all_data_out.bin" , "rb")
    fin = open("../usefulData/all_data_in.bin" , "rb")
    data_in = pickle.load(fin)
    data_out = pickle.load(fout)
    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, test_size=0.01, random_state=42)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return (X_train,y_train), (X_test, y_test)

def ssim(y_true, y_pred):
    return tf.image.ssim( y_true, y_pred, max_val= 10)


(train_images, train_labels), (test_images, test_labels) = get_data()

train_images = train_images.reshape((train_images.shape[0], 19, 19, 1))
test_images = test_images.reshape((test_images.shape[0], 19, 19, 1))

train_labels = train_labels.reshape((train_labels.shape[0], 225))
test_labels = test_labels.reshape((test_labels.shape[0], 225))

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(19, 19, 1)))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))

#model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(225, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

#x = model.predict(test_images[0].reshape((1,19,19,1)))
#print(x.shape)
model.fit(train_images, train_labels, epochs=30)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

x = model.predict(test_images[0].reshape((1,19,19,1)))
x = x.reshape((15,15))
print(x)
plt.colorbar()
plt.imshow(x)
plt.show()
plt.imshow(test_images[0].reshape((19,19)))
plt.show()