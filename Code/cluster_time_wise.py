import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.measure import compare_ssim as ssim


def met(im1,im2):
    return  1 - ssim(im1,im2)

time = 10
for time in range(0,1):
    fout = open("../usefulData/all_data_out.bin" , "rb")
    data_out = pickle.load(fout)
    data7_out = []

    for i in range(365):
        data7_out.append(np.array(data_out[24*i + time - 1]).reshape(225))

    #print(data7_out)
    with open("../usefulData/Hourly/" + str(time) + "data.bin", "wb") as f:
        pickle.dump(data7_out, f)

    data7_out = np.array(data7_out)

    #kmeans = km.KMeans(n_clusters=2, random_state=0).fit(data7_out)

    for eps in range(1,50, 2 ):
        kmeans = DBSCAN(eps = eps/100  , min_samples=10, metric= met ).fit(data7_out)
        print(eps)
        print(kmeans.labels_)

    data = { 1:{ 0:0, 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0} ,0: {0:0 , 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0}}

    print(kmeans.labels_)
    for l in range(len(kmeans.labels_)):
        #print(l)
        if kmeans.labels_[l] != -1:
            data[kmeans.labels_[l]][l%7] += 1

    print(data)

