import numpy as np
import pickle
import numpy as np
#from sklearn.cluster import DBSCAN
from skimage.measure import compare_ssim as ssim

class KMeans:
    def __init__(self, k= 3, tol= 0.1, max_iter=3000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids ={}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifciations = {}

            for i in range(self.k):
              self.classifciations[i] =[]

            for featureset in data:
                distances = [(1-ssim(featureset ,self.centroids[centroid])) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifciations[classification].append(featureset)

        prev_centroids = dict(self.centroids)

        for classification in self.classifciations:
            #pass
            self.centroids[classification] =np.average(self.classifciations[classification], axis = 0)

        optimized = True

        for c in self.centroids:
            original_centroid = prev_centroids[c]
            current_centroid = self.centroids[c]
            if ( 1 - ssim(current_centroid , original_centroid) ) * 10.0 < self.tol:
                print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                print("Optimized")
                optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [( 1 - ssim(data ,self.centroids[centroid])) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



for time in range(0,24):
    fout = open("../usefulData/all_data_out.bin" , "rb")
    data_out = pickle.load(fout)
    data7_out = []

    for i in range(365):
        data7_out.append(np.array(data_out[24*i + time - 1]).reshape(225))

    #print(data7_out)
    with open("../usefulData/Hourly/" + str(time) + "data.bin", "wb") as f:
        pickle.dump(data7_out, f)

    data7_out = np.array(data7_out)

    clf = KMeans(k=2)
    clf.fit(data7_out)

    #kmeans = DBSCAN(eps = 0.1  , min_samples=10, metric= met ).fit(data7_out)

    label = np.zeros(len(data7_out))
    for i in range(len(data7_out)):
        label[i] = clf.predict(data7_out[i])

    #print(label)
    data = {2:{ 0:0, 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0}, 1:{ 0:0, 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0} ,0: {0:0 , 1:0 , 2:0 ,3:0 , 4:0, 5:0 , 6:0}}


    #print(kmeans.labels_)
    for l in range(len(label)):
        #print(l)
        if label[l] != -1:
            data[label[l]][l%7] += 1

    print(data)