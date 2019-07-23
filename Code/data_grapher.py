import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
model = "CNN"
time_inp = 1
denselayers = 5
epoch = 50
fout = open("../usefulData/"+str(time_inp)+"_inp_data_out7am_red.bin" , "rb")
fin = open("../usefulData/"+str(time_inp)+"_inp_data_in7am_red.bin" , "rb")
data_in = pickle.load(fin)
data_out = pickle.load(fout)
X_train, X_test, y_train, label = train_test_split(data_in, data_out, test_size=0.1, random_state=20)
fout.close()
fin.close()
with open("../usefulData/pred/"+str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochs"+model , "rb") as f:
    pred = pickle.load(f)


label = np.array(label)
pred = np.array(pred.reshape((len(pred),13,13)))

sarr = np.zeros(len(label))
marr = np.zeros(len(label))
for i in range(len(sarr)):
    sarr[i] = ssim(label[i],pred[i])

for i in range(len(label)):
    marr[i] = np.mean( np.square((label[i] - pred[i])))



print(sarr)
print(np.mean(sarr))
print(np.min(sarr))
print(np.max(sarr))

print(marr)
print(np.mean(marr))
print(np.min(marr))
print(np.max(marr))

label = label.flatten()
pred = pred.flatten()

print(len(label))
print(len(pred))
plt.scatter(pred, label, s=1)

#plt.title("Extract Number Root ")

    # Set x, y label text.
plt.ylabel("Label")
plt.xlabel("Pred")
x3 = np.linspace(11, 35, 500)
x2 = np.linspace(6,10,50)
x1 = np.linspace(0,5,50)
x = np.linspace(0,35,1000)
plt.plot(x, x + 0, linestyle='solid', color='g')
plt.plot(x1, x1 + 2, linestyle='solid', color='r')
plt.plot(x1, x1 - 2, linestyle='solid', color='r')
plt.plot(x2, x2 + 3, linestyle='solid', color='r')
plt.plot(x2, x2 - 3, linestyle='solid', color='r')
plt.plot(x3, 0.8*x3 , linestyle='solid', color='r')
plt.plot(x3, 1.2*x3 , linestyle='solid', color='r')
plt.show()