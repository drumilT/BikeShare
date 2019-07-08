import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import  date as dtf
import holidays
import pickle

us_holidays = holidays.US()

scale = 1

def get_holiday(now):
    if now in us_holidays:
        return 1*scale
    else:
        return 0


def get_season(now):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [(0.25, 'winter', (dtf(Y, 1, 1), dtf(Y, 3, 20))),
               (0.5, 'spring', (dtf(Y, 3, 21), dtf(Y, 6, 20))),
               (0.75, 'summer', (dtf(Y, 6, 21), dtf(Y, 9, 22))),
               (1, 'autumn', (dtf(Y, 9, 23), dtf(Y, 12, 20))),
               (0.25, 'winter', (dtf(Y, 12, 21), dtf(Y, 12, 31)))]

    if isinstance(now, datetime):
        now = now.dtf()
    now = now.replace(year=Y)
    return next( scale*season_num for season_num, season, (start, end) in seasons
                if start <= now <= end)


def get_day(now):
    return  ((now.weekday()) + 1) / 7


def make_st_num_dict():
    bij = {31100: 3, 31101: 8, 31102: 4, 31103: 4, 31104: 10, 31105: 4,
          31106: 10,
          31107: 4, 31108: 11, 31201: 2, 31202: 8, 31203: 1, 31204: 3,
          31205: 3,
          31400: 4, 31401: 4, 31502: -1, 31600: 6, 31602: 4, 31305: 0,
          31206: 3,
          31500: -1, 31111: 8, 31207: 8, 31110: 10, 31109: 8, 31200: 2,
          31603: 9,
          31212: 2, 31213: 2, 31604: 6, 31214: 2, 31503: 9, 31302: 0,
          31402: 12,
          31216: 1, 31217: 5, 31215: 7, 31220: 3, 31218: 11, 31219: 5,
          31211: 7,
          31221: 2, 31620: 6, 31222: 1, 31223: 6, 31112: 10, 31224: 2,
          31225: 7,
          31609: 11, 31226: 0, 31227: 1, 31228: 6, 31505: 9, 31229: 2,
          31230: 1,
          31231: 1, 31232: 6, 31233: 2, 31234: 2, 31621: 6, 31235: 3, 31237: 7,
          31624: 6, 31266: 1, 31304: 0, 31238: 1, 31240: 3, 31262: 1, 31260: 3,
          31261: 3, 31113: 10, 31239: 2, 31241: 1, 31242: 3, 31243: 11,
          31244: 11,
          31245: 8, 31404: 12, 31506: 9, 31115: 4, 31116: 10, 31307: 0,
          31246: 7,
          31263: 1, 31507: 9, 31247: 5, 31248: 5, 31264: 6, 31249: 5,
          31250: 2,
          31251: 1, 31252: 3, 31253: 2, 31254: 1, 31255: 7, 31256: 1, 31257: 3,
          31258: 3, 31259: 3, 31265: 6, 31114: 10, 31405: 12, 31406: 12,
          31312: 7,
          31267: 2, 31117: 4, 31509: 9, 31268: 8, 31270: 6, 31118: 9,
          31513: 9,
          31271: 11, 31272: 11, 31633: 5, 31514: -1, 31119: 8, 31120: 8,
          31121: 10, 31636: 6, 31273: 5, 31637: 6, 31638: 6, 31515: -1,
          31274: 1,
          31275: 7, 31276: 1, 31277: 3, 31278: 2, 31279: 3, 31522: 9,
          31293: 7,
          31280: 8, 31281: 6, 31122: 4, 31282: 2, 31283: 1, 31519: 9,
          31284: 3,
          31285: 2, 31123: 4, 31286: 1, 31287: 5, 31288: 11, 31289: 3,
          31290: 5,
          31642: 6, 31124: 4, 31291: 1, 31292: 3, 31125: 8, 31294: 11,
          31295: 7,
          31296: 10, 31297: 7, 31298: 1, 31126: 4, 31299: 2, 31321: 5,
          31127: 3,
          31128: 7, 31129: 1, 31646: 5, 31523: 9, 31649: 4, 31651: 4,
          31417: 12,
          31653: 6, 31655: 6, 31323: 10, 31418: -1, 31324: 2}
    return bij

bij = make_st_num_dict()
c_date = 0
c_hour = 1
c_st_st = 2
c_en_st = 3
num_clusture = 13
time_inp = 1

def give_date_time(arr):
     date = int(arr[8:10])
     hour = int(arr[11:13])
     return date, hour

def get_months_data(month) :
    ext = "-capitalbikeshare-tripdata.csv"
    df = pd.read_csv("../Data/" + str(201800 + month) + ext)
    data = np.zeros((df.shape[0], 4))
    df.columns = df.columns.str.strip().str.lower().str.replace(' ',  '_').str.replace('(', '').str.replace(')', '')
    for i in range(data.shape[0]):
        data[i, c_date], data[i, c_hour] = give_date_time(df.start_date[i])
        data[i, c_st_st] = df.start_station_number[i]
        data[i, c_en_st] = df.end_station_number[i]
    #data = np.sort(data, axis = c_hour, kind = 'mergesort')
    #data = np.sort(data, axis = c_date, kind = 'mergesort')
    return data

def date_od_arr_in(s_data, bij, num_clusture, day, season , holiday):
    od_arr = np.zeros((24, (num_clusture + 4) , (num_clusture+4) ))

    od_arr[:, num_clusture + 1, : ] = day
    od_arr[:, num_clusture + 2, : ] = season
    od_arr[:, num_clusture + 3, : ] = holiday
    for i in range(24):
        od_arr[i,num_clusture, :] = i/23

    for i in range(s_data.shape[0]):
        start = bij.setdefault(s_data[i,c_st_st], -1)
        end = bij.setdefault(s_data[i,c_en_st], -1)
        hour = int(s_data[i,c_hour])
        #print(hour, start, end)
        if(start != -1 and end != -1):
            od_arr[hour, start, end] += scale

    #od_arr = np.reshape(od_arr, (24, num_clusture * num_clusture))

    return od_arr

def date_od_arr_out(s_data, bij, num_clusture):
    od_arr = np.zeros((24, num_clusture , num_clusture ))


    for i in range(s_data.shape[0]):
        start = bij.setdefault(s_data[i,c_st_st], -1)
        end = bij.setdefault(s_data[i,c_en_st], -1)
        hour = int(s_data[i,c_hour])
        #print(hour, start, end)
        if(start != -1 and end != -1):
            od_arr[hour, start, end] += scale

    return od_arr

fout = open("../usefulData/"+str(time_inp)+"_inp_data_out_red.bin" , "wb")
fin = open("../usefulData/"+str(time_inp)+"_inp_data_in_red.bin" , "wb")

counta = 0
countb = 1
year_data_in = []
year_data_out = []
lst = [None] * time_inp

for month in range(1,13):
    st = time.time()
    print(month)
    print("Loading Data")
    data = get_months_data(month = month)
    print("Processing", time.time() - st)


    for date in range(1, 32):
        s_data = data[data[:, c_date] == date].copy()
        if(s_data.shape[0] == 0):
            continue
        timest = dtf(2018, month, date)
        season = get_season(timest)
        day = get_day(timest)
        holiday = get_day(timest)

        for l in date_od_arr_out(s_data=s_data, bij=bij, num_clusture=num_clusture ):
            if counta >= time_inp:
                year_data_out.append(l)
            counta += 1

        for l in date_od_arr_in(s_data=s_data, bij=bij, num_clusture=num_clusture,
                            season=season, day=day, holiday=holiday):
            lst.pop(0)
            lst.append(l)
            if countb >= time_inp:
                stacked_inp = None
                for l2 in lst:
                    l3 = np.array(l2)
                    if isinstance(stacked_inp, np.ndarray):
                        print(stacked_inp.shape)
                        stacked_inp = np.vstack((stacked_inp,l3))
                    else:
                        stacked_inp = l3
                print(stacked_inp.shape)
                year_data_in.append(stacked_inp.tolist())

            countb += 1
    print(time.time() - st)



pickle.dump( year_data_out, fout)
year_data_in.pop()
pickle.dump(year_data_in, fin)

print(len(year_data_out))
print(len(year_data_in))

fout.close()
fin.close()


fout = open("../usefulData/"+str(time_inp)+"_inp_data_out_red.bin" , "rb")
fin = open("../usefulData/"+str(time_inp)+"_inp_data_in_red.bin" , "rb")

fout7 = open("../usefulData/"+str(time_inp)+"_inp_data_out7_red.bin" , "wb")
fin7 = open("../usefulData/"+str(time_inp)+"_inp_data_in7_red.bin" , "wb")


data_in = pickle.load(fin)

data7_in = []
for i in range(365):
    for k in range(7,21):
        if i <= 1 :
            print(data_in[24*i + k - time_inp][-4][0])
        data7_in.append(data_in[24*i + k - time_inp])

data7_inarr = np.array(data7_in)
print(data7_inarr.shape)

pickle.dump(data7_in, fin7)

data_out = pickle.load(fout)
data7_out = []

for i in range(365):
    for k in range(7,21):
        data7_out.append(data_out[24*i + k - time_inp])

data7_outarr = np.array(data7_out)
print(data7_outarr.shape)

pickle.dump(data7_out, fout7)

