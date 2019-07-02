from __future__ import absolute_import, division, print_function, unicode_literals

#pip install -q tensorflow-gpu==2.0.0-beta1
import tensorflow as tf
import pickle

from tensorflow.keras import datasets, layers, models

def get_data():
    fout = open("../usefulData/all_data_out.bin" , "rb")
    fin = open("../usefulData/all_data_in.bin" , "rb")
    data_in = pickle.load(fin)
    data_out = pickle.load(fout)
    


(train_images, train_labels), (test_images, test_labels) = get_data()

#train_images = train_images.reshape((60000, 28, 28, 1))
#test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
