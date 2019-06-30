import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import  date as dtf
import holidays
import pickle

us_holidays = holidays.US()


def get_holiday(now):
    if now in us_holidays:
        return 1
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
    return next(season_num for season_num, season, (start, end) in seasons
                if start <= now <= end)


def get_day(now):
    return  ((now.weekday()) + 1) / 7


def make_st_num_dict():
    bij = {31100: 4, 31101: 9, 31102: 5, 31103: 5, 31104: 12, 31105: 5,
          31106: 12,
          31107: 5, 31108: 13, 31201: 2, 31202: 9, 31203: 1, 31204: 4,
          31205: 4,
          31400: 5, 31401: 5, 31502: 10, 31600: 7, 31602: 5, 31305: 0,
          31206: 4,
          31500: 10, 31111: 9, 31207: 9, 31110: 12, 31109: 9, 31200: 2,
          31603: 11,
          31212: 2, 31213: 2, 31604: 7, 31214: 2, 31503: 11, 31302: 0,
          31402: 14,
          31216: 1, 31217: 6, 31215: 8, 31220: 4, 31218: 13, 31219: 6,
          31211: 8,
          31221: 2, 31620: 7, 31222: 1, 31223: 7, 31112: 12, 31224: 2,
          31225: 8,
          31609: 13, 31226: 0, 31227: 1, 31228: 7, 31505: 11, 31229: 2,
          31230: 1,
          31231: 1, 31232: 7, 31233: 2, 31234: 2, 31621: 7, 31235: 4, 31237: 8,
          31624: 7, 31266: 1, 31304: 0, 31238: 1, 31240: 4, 31262: 1, 31260: 4,
          31261: 4, 31113: 12, 31239: 2, 31241: 1, 31242: 4, 31243: 13,
          31244: 13,
          31245: 9, 31404: 14, 31506: 11, 31115: 5, 31116: 12, 31307: 0,
          31246: 8,
          31263: 1, 31507: 11, 31247: 6, 31248: 6, 31264: 7, 31249: 6,
          31250: 2,
          31251: 1, 31252: 4, 31253: 2, 31254: 1, 31255: 8, 31256: 1, 31257: 4,
          31258: 4, 31259: 4, 31265: 7, 31114: 12, 31405: 14, 31406: 14,
          31312: 8,
          31267: 2, 31117: 5, 31509: 11, 31268: 9, 31270: 7, 31118: 11,
          31513: 11,
          31271: 13, 31272: 13, 31633: 6, 31514: 10, 31119: 9, 31120: 9,
          31121: 12, 31636: 7, 31273: 6, 31637: 7, 31638: 7, 31515: 3,
          31274: 1,
          31275: 8, 31276: 1, 31277: 4, 31278: 2, 31279: 4, 31522: 11,
          31293: 8,
          31280: 9, 31281: 7, 31122: 5, 31282: 2, 31283: 1, 31519: 11,
          31284: 4,
          31285: 2, 31123: 5, 31286: 1, 31287: 6, 31288: 13, 31289: 4,
          31290: 6,
          31642: 7, 31124: 5, 31291: 1, 31292: 4, 31125: 9, 31294: 13,
          31295: 8,
          31296: 12, 31297: 8, 31298: 1, 31126: 5, 31299: 2, 31321: 6,
          31127: 4,
          31128: 8, 31129: 1, 31646: 6, 31523: 11, 31649: 5, 31651: 5,
          31417: 14,
          31653: 7, 31655: 7, 31323: 12, 31418: 3, 31324: 2}
    return bij

bij = make_st_num_dict()
c_date = 0
c_hour = 1
c_st_st = 2
c_en_st = 3
num_clusture = 15

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
            od_arr[hour, start, end] += 1

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
            od_arr[hour, start, end] += 1

    return od_arr

fout = open("../Data/all_data_out.bin" , "wb")
fin = open("../Data/all_data_in.bin" , "wb")

count = False
for month in range(1,13):
    st = time.time()
    print(month)
    print("Loading Data")
    data = get_months_data(month = month)
    print("Processing", time.time() - st)
    months_data_in = []
    months_data_out = []
    for date in range(1, 32):
        s_data = data[data[:, c_date] == date].copy()
        if(s_data.shape[0] == 0):
            continue
        timest = dtf(2018, month, date)
        season = get_season(timest)
        day = get_day(timest)
        holiday = get_day(timest)
        if count:
            months_data_out.append(date_od_arr_out(s_data=s_data, bij=bij, num_clusture=num_clusture ))

        months_data_in.append(
            date_od_arr_in(s_data=s_data, bij=bij, num_clusture=num_clusture,
                            season=season, day=day, holiday=holiday))
        count = True

    pickle.dump( months_data_out, fout)
    if month == 12:
        months_data_in.pop()

    pickle.dump(months_data_in, fin)
    print(time.time() - st)

fout.close()
fin.close()