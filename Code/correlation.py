import pickle
import numpy as np
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt


## draws ssim matrix for each hour of all daily data
##essentially 24 ssim matrices


def compare_image(im1,im2):
    return ssim(im1,im2)

for i in range(24):
    f = open("../usefulData/Hourly/"+str(i)+"data.bin","rb")
    data = pickle.load(f)
    data = np.array(data)
    simarr = np.zeros((365,365))
    for x in range(365):
        for y in range(365):
            if simarr[x][y]==0 :
                X_data = data[x]
                Y_data = data[y]
                val = compare_image(X_data,Y_data)
                simarr[x][y] = val
                simarr[y][x] = val

    plt.imshow(simarr)
    plt.suptitle(str(i*100)+"hours ssim matrix")
    #plt.show()
    if i == 0:
        plt.colorbar()
    plt.savefig("../Graph/Corr/" + str(i) + 'Hour-graph.png')
