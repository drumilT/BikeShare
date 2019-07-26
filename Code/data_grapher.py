import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from error_function import travel_data_accuracy, travel_data_error

#will print the travel_data_accuracy, travel_data_error, mse and ssim for the predicted data also will store the label
# vs pred plot



label_dim = 13
model = "CNN_new" ## CNN/ ANN/ CNN_new
time_inp = 3
denselayers = 11
epoch = 100


###dont edit array anmes below
acc = [] # stores the list of mean travel_data_accuracy for mulitple run function calls
ssima = [] # stores the list of mean ssim for mulitple run function calls
mse = [] # stores the list of mean mse for mulitple run function calls
error = []  # stores the list of mean travel_data_error for mulitple run function calls
good= [] # stores tuples of good control params defined by contraints on other measures


def run():
    print(str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochs"+model)

    fout = open("../usefulData/"+str(time_inp)+"_inp_data_out7am_red_mem.bin" , "rb")
    fin = open("../usefulData/"+str(time_inp)+"_inp_data_in7am_red_mem.bin" , "rb")
    data_in = pickle.load(fin)
    data_out = pickle.load(fout)
    X_train, X_test, y_train, label = train_test_split(data_in, data_out, test_size=0.1, random_state=100)
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



    #print(sarr)
    print(np.mean(sarr))
    ssima.append( np.mean(sarr))
    #print(np.min(sarr))
    #print(np.max(sarr))

    #print(marr)
    print(np.mean(marr))
    mse.append(np.mean(marr))
    #print(np.min(marr))
    #print(np.max(marr))

    #print(len(label))
    #print(len(pred))
    label = np.array(label.reshape((len(pred),169)))
    pred = np.array(pred.reshape((len(pred),169)))
    arre = []
    arra = []
    for t, s in zip(pred, label):
        # print(count)
        arre.append(travel_data_error(s, t))
        arra.append(travel_data_accuracy(s, t))
        t = np.around(t)
        s = np.around(s)
        t = t.reshape((label_dim, label_dim))
        s = s.reshape((label_dim, label_dim))

    #print(arre)
    print(np.mean(np.array(arre)))
    error.append(np.mean(np.array(arre)))
    #print(arra)
    print(np.mean(np.array(arra)))


    acc.append(np.mean(np.array(arra)))


    if np.mean(np.array(arra)) >0.9 and np.mean(np.array(arre)) < 2:
        good.append((time_inp,denselayers,epoch))

    label = np.around(label.flatten())
    pred = np.around(pred.flatten())

    plt.scatter(pred, label, s=1)

    plt.title(str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochs"+model)
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
    plt.savefig("../Graph/pred_label/CNN/"+str(time_inp)+"hr_inp"+str(denselayers)+"dense_layers"+str(epoch)+"epochs"+model)
    plt.close()


## evrything below used for tuning purposes not related to main code
def rough_tuning_check():
    time_inps = [1,3,5]
    epochs = [50,100,500,1000]
    denselayerss =[5,7,9,11,13,15,17,20]
    global time_inp,epoch,denselayers, model
    model = "CNN"
    count = 0
    for t in time_inps:
        for e in epochs:
            for d in denselayerss:
                if e >= d*5:
                    time_inp = t
                    epoch = e
                    denselayers = d
                    if count < 74:
                        run()
                        count+=1

## used for tuning purposes not related to main code
def refined_tuning_checkup():
    finely_tuned = [(1, 7, 100), (1, 9, 100), (1, 17, 500), (3, 5, 50), (3, 9, 100), (3, 11, 100), (3, 13, 100), (3, 15, 100), (3, 17, 500), (5, 11, 100), (5, 13, 100)]

    global epoch, time_inp, denselayers, model
    model = "CNN"
    for time,dense,epo in finely_tuned:
        jump = int(0.1 * epo)
        epochs = [epo]
        for i in range(1,4):
            t = epo - i*jump
            s = epo + i*jump
            epochs.append(t)
            epochs.append(s)
        #print(epochs)
        for e in epochs:
            time_inp = time
            epoch = e
            denselayers = dense
            run()

def  cnn_new_checkup():
    global epoch, time_inp, denselayers,model
    model ="CNN_new"
    epochs = [10, 30, 50, 70, 100, 300, 700, 1000]
    for e in epochs:
        epoch = e
        time_inp = 1
        denselayers = 5
        run()


run()
#refined_tuning_checkup()
#cnn_new_checkup()
print("##################################")
print(np.max(acc))
print(np.min(error))
print(np.min(mse))
print(np.min(ssima))

print(good)