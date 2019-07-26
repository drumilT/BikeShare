
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date, datetime
import holidays
import pickle
# starting
##older but reliable code to get hourly OD graph can be used to cross check if hourly data fast is working correctly
##slower code



dd = 1
m = 1
hr = 0
time_period = 1  # in hours
num_of_clusters = 7
df = pd.read_csv("../Data/station_information_Mar_2019.csv")
sta = df.name
bij = {31100: 4, 31101: 9, 31102: 5, 31103: 5, 31104: 12, 31105: 5, 31106: 12,
       31107: 5, 31108: 13, 31201: 2, 31202: 9, 31203: 1, 31204: 4, 31205: 4,
       31400: 5, 31401: 5, 31502: 10, 31600: 7, 31602: 5, 31305: 0, 31206: 4,
       31500: 10, 31111: 9, 31207: 9, 31110: 12, 31109: 9, 31200: 2, 31603: 11,
       31212: 2, 31213: 2, 31604: 7, 31214: 2, 31503: 11, 31302: 0, 31402: 14,
       31216: 1, 31217: 6, 31215: 8, 31220: 4, 31218: 13, 31219: 6, 31211: 8,
       31221: 2, 31620: 7, 31222: 1, 31223: 7, 31112: 12, 31224: 2, 31225: 8,
       31609: 13, 31226: 0, 31227: 1, 31228: 7, 31505: 11, 31229: 2, 31230: 1,
       31231: 1, 31232: 7, 31233: 2, 31234: 2, 31621: 7, 31235: 4, 31237: 8,
       31624: 7, 31266: 1, 31304: 0, 31238: 1, 31240: 4, 31262: 1, 31260: 4,
       31261: 4, 31113: 12, 31239: 2, 31241: 1, 31242: 4, 31243: 13, 31244: 13,
       31245: 9, 31404: 14, 31506: 11, 31115: 5, 31116: 12, 31307: 0, 31246: 8,
       31263: 1, 31507: 11, 31247: 6, 31248: 6, 31264: 7, 31249: 6, 31250: 2,
       31251: 1, 31252: 4, 31253: 2, 31254: 1, 31255: 8, 31256: 1, 31257: 4,
       31258: 4, 31259: 4, 31265: 7, 31114: 12, 31405: 14, 31406: 14, 31312: 8,
       31267: 2, 31117: 5, 31509: 11, 31268: 9, 31270: 7, 31118: 11, 31513: 11,
       31271: 13, 31272: 13, 31633: 6, 31514: 10, 31119: 9, 31120: 9,
       31121: 12, 31636: 7, 31273: 6, 31637: 7, 31638: 7, 31515: 3, 31274: 1,
       31275: 8, 31276: 1, 31277: 4, 31278: 2, 31279: 4, 31522: 11, 31293: 8,
       31280: 9, 31281: 7, 31122: 5, 31282: 2, 31283: 1, 31519: 11, 31284: 4,
       31285: 2, 31123: 5, 31286: 1, 31287: 6, 31288: 13, 31289: 4, 31290: 6,
       31642: 7, 31124: 5, 31291: 1, 31292: 4, 31125: 9, 31294: 13, 31295: 8,
       31296: 12, 31297: 8, 31298: 1, 31126: 5, 31299: 2, 31321: 6, 31127: 4,
       31128: 8, 31129: 1, 31646: 6, 31523: 11, 31649: 5, 31651: 5, 31417: 14,
       31653: 7, 31655: 7, 31323: 12, 31418: 3, 31324: 2}
ext = "-capitalbikeshare-tripdata.csv"
us_holidays = holidays.US()


def get_holiday(now):
    if now in us_holidays:
        return 1
    else:
        return 0


def get_season(now):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [(0.25, 'winter', (date(Y, 1, 1), date(Y, 3, 20))),
               (0.5, 'spring', (date(Y, 3, 21), date(Y, 6, 20))),
               (0.75, 'summer', (date(Y, 6, 21), date(Y, 9, 22))),
               (1, 'autumn', (date(Y, 9, 23), date(Y, 12, 20))),
               (0.25, 'winter', (date(Y, 12, 21), date(Y, 12, 31)))]

    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season_num for season_num, season, (start, end) in seasons
                if start <= now <= end)


def biject(key):
    return bij.setdefault(key, 173)


count = 1
data = 0

f = open("../everything.txt", "ab")

for i in range(m, 13):
    print(i)
    df = pd.read_csv("../Data/" + str(201800 + i) + ext)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ',
                                                                '_').str.replace(
        '(', '').str.replace(')', '')
    df["start_date"] = df.start_date.apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    # df = df[ (df.start_station_number in bij.keys()) & (df.end_station_number in bij.keys())]
    for k in range(dd, 32):

        df["day"] = df.start_date.apply(lambda x: x.day)
        dfd = df[df.day == k]
        if len(dfd) != 0:
            day = ((datetime(2018, i, k).weekday()) + 1) / 7
            for j in range(int(24 / time_period)):
                dfd["hour"] = dfd.start_date.apply(lambda x: x.hour)
                dfdh = dfd[(dfd.hour >= (time_period * j)) & (
                            dfd.hour < (time_period * (j + 1)))]
                # print(len(dfdh))
                #dfdh["start_sta_num_key"] =
                x = dfdh.start_station_number
                y = dfdh.end_station_number
                x = x.apply(lambda key: bij.setdefault(key, 173))
                y = y.apply(biject)
                #print(x)
                #print(
                 #   "##############################################################")
                #print(y)
                hist2D = plt.hist2d(list(y), list(x), np.array(range(16)))
                # print(hist2D[0])
                arr = np.array(hist2D[0])
                arr = arr.flatten()
                arr = np.append(arr, (j / 24))
                arr = np.append(arr, day)
                arr = np.append(arr, i)
                arr = np.append(arr, get_season(datetime(2018, i, k)))
                arr = np.append(arr, get_holiday(date(2018, i, k)))
                pickle.dump(arr,f)
                # if count == 1:
                #   plt.colorbar()#plt.pco   lor(hist2D)
                # plt.show()
                # plt.savefig("../Data/Graph/"+str(num_of_clusters)+"-Clusters/"+str(time_period) + "-hour/"+ str(2018000000 + i*10000 + k*100 + time_period*j)+'graph.png')
                # count=0
                #print(data)
                #print("################################")

f.close()