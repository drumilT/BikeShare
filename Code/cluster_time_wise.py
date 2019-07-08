import pickle
import numpy as np
from sklearn.cluster.k_means_ import KMeans
time = 10
for time in range(0,23):
    fout = open("../usefulData/all_data_out.bin" , "rb")
    data_out = pickle.load(fout)
    data7_out = []

    for i in range(365):
        data7_out.append(np.array(data_out[24*i + time - 1]).reshape(225))

    #print(data7_out)
    with open("../usefulData/" + str(time) + "data.bin", "wb") as f:
        pickle.dump(data7_out, f)

    data7_out = np.array(data7_out)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(data7_out)


    data = { 1:{ 0:0, 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0} ,0: {0:0 , 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0}}

    for l in range(len(kmeans.labels_)):
        data[kmeans.labels_[l]][l%7] += 1

    print(data)

