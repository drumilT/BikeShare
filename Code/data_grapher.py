import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

time_inp = 5
denselayers = 5
epoch = 500

fout = open("../usefulData/"+str(time_inp)+"_inp_data_out7am_red.bin" , "rb")
fin = open("../usefulData/"+str(time_inp)+"_inp_data_in7am_red.bin" , "rb")
data_in = pickle.load(fin)
data_out = pickle.load(fout)
X_train, X_test, y_train, label = train_test_split(data_in, data_out, test_size=0.1, random_state=200)

with open("../usefulData/pred/"+str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochs" , "rb") as f:
    pred = pickle.load(f)


label = np.array(label)
pred = np.array(pred)

label = label.flatten()
pred = pred.flatten()

print(len(label))
print(len(pred))
plt.scatter(pred, label, s=1)

#plt.title("Extract Number Root ")

    # Set x, y label text.
plt.ylabel("Label")
plt.xlabel("Pred")
x = np.linspace(0, 60, 1000)
plt.plot(x, x + 0, linestyle='solid', color='g')
plt.plot(x, x + 10, linestyle='solid', color='r')
plt.plot(x, x - 10, linestyle='solid', color='r')
plt.show()