import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import  date as dtf
import holidays
import pickle


###Control Variables####
us_holidays = holidays.US()
scale = 1
c_date = 0
c_hour = 1
c_st_st = 2
c_en_st = 3
num_clusture = 13
time_inp = 3
new_bij = {31305: 0, 31302: 1, 31226: 2, 31304: 3, 31307: 4, 31203: 5, 31216: 6, 31222: 7, 31227: 8, 31230: 9, 31231: 10, 31266: 11, 31238: 12, 31262: 13, 31241: 14, 31263: 15, 31251: 16, 31254: 17, 31256: 18, 31274: 19, 31276: 20, 31283: 21, 31286: 22, 31291: 23, 31298: 24, 31129: 25, 31201: 26, 31200: 27, 31212: 28, 31213: 29, 31214: 30, 31221: 31, 31224: 32, 31229: 33, 31233: 34, 31234: 35, 31239: 36, 31250: 37, 31253: 38, 31267: 39, 31278: 40, 31282: 41, 31285: 42, 31299: 43, 31324: 44, 31100: 45, 31204: 46, 31205: 47, 31206: 48, 31220: 49, 31235: 50, 31240: 51, 31260: 52, 31261: 53, 31242: 54, 31252: 55, 31257: 56, 31258: 57, 31259: 58, 31277: 59, 31279: 60, 31284: 61, 31289: 62, 31292: 63, 31127: 64, 31102: 65, 31103: 66, 31105: 67, 31107: 68, 31400: 69, 31401: 70, 31602: 71, 31115: 72, 31117: 73, 31122: 74, 31123: 75, 31124: 76, 31126: 77, 31649: 78, 31651: 79, 31217: 80, 31219: 81, 31247: 82, 31248: 83, 31249: 84, 31633: 85, 31273: 86, 31287: 87, 31290: 88, 31321: 89, 31646: 90, 31600: 91, 31604: 92, 31620: 93, 31223: 94, 31228: 95, 31232: 96, 31621: 97, 31624: 98, 31264: 99, 31265: 100, 31270: 101, 31636: 102, 31637: 103, 31638: 104, 31281: 105, 31642: 106, 31653: 107, 31655: 108, 31215: 109, 31211: 110, 31225: 111, 31237: 112, 31246: 113, 31255: 114, 31312: 115, 31275: 116, 31293: 117, 31295: 118, 31297: 119, 31128: 120, 31101: 121, 31202: 122, 31111: 123, 31207: 124, 31109: 125, 31245: 126, 31268: 127, 31119: 128, 31120: 129, 31280: 130, 31125: 131, 31603: 132, 31503: 133, 31505: 134, 31506: 135, 31507: 136, 31509: 137, 31118: 138, 31513: 139, 31522: 140, 31519: 141, 31523: 142, 31104: 143, 31106: 144, 31110: 145, 31112: 146, 31113: 147, 31116: 148, 31114: 149, 31121: 150, 31296: 151, 31323: 152, 31108: 153, 31218: 154, 31609: 155, 31243: 156, 31244: 157, 31271: 158, 31272: 159, 31288: 160, 31294: 161, 31402: 162, 31404: 163, 31405: 164, 31406: 165, 31417: 166}
input_len = 167
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


def give_date_time(arr):
     date = int(arr[8:10])
     hour = int(arr[11:13])
     return date, hour


def get_months_data(month) :
    ext = "-capitalbikeshare-tripdata.csv"
    df = pd.read_csv("../Data/" + str(201800 + month) + ext)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ',  '_').str.replace('(', '').str.replace(')', '')
    df = df[df.member_type=="Member"]
    data = np.zeros((df.shape[0], 4))
    for i in range(data.shape[0]):
        #print(df.index[i])
        data[i, c_date], data[i, c_hour] = give_date_time(df.start_date[df.index[i]])
        data[i, c_st_st] = df.start_station_number[df.index[i]]
        data[i, c_en_st] = df.end_station_number[df.index[i]]
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

def date_od_arr_in_new(s_data, bij, num_clusture, day, season , holiday):
    od_arr = np.zeros((24, input_len + 1, input_len ))

    od_arr[:, input_len,int(input_len/4) : 2*int(input_len/4) ] = day
    od_arr[:, input_len, 2*int(input_len/4):3*int(input_len/4) ] = season
    od_arr[:, input_len, 3*int(input_len/4): 4*int(input_len/4)] = holiday
    for i in range(24):
        od_arr[i,input_len, : int(input_len/4)] = i/23

    for i in range(s_data.shape[0]):
        start = new_bij.setdefault(s_data[i,c_st_st], -1)
        end = new_bij.setdefault(s_data[i,c_en_st], -1)
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

def create_166_input_new():

    fin = open("../usefulData/"+str(time_inp)+"_inp_data_in_166_mem.bin" , "wb")
    year_data_in = []
    year_data_out = []
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

            timet = [6,7,8]
            counta = 0
            for l in date_od_arr_in_new(s_data=s_data, bij=bij, num_clusture=num_clusture,
                                season=season, day=day, holiday=holiday):
                if counta in timet:
                    year_data_in.append(l)
                counta += 1
        print(time.time() - st)

    pickle.dump(year_data_in, fin)
    print(len(year_data_in))
    fin.close()

def create_total_input():
    fout = open("../usefulData/"+str(time_inp)+"_inp_data_out_red_mem.bin" , "wb")
    fin = open("../usefulData/"+str(time_inp)+"_inp_data_in_red_mem.bin" , "wb")

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


def create_7_to_8pm_data():
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


def create_5_to_10am_data():
    fout = open("../usefulData/" + str(time_inp) + "_inp_data_out_red_mem.bin", "rb")
    fin = open("../usefulData/" + str(time_inp) + "_inp_data_in_red_mem.bin", "rb")

    fout7 = open("../usefulData/" + str(time_inp) + "_inp_data_out7am_red_mem.bin", "wb")
    fin7 = open("../usefulData/" + str(time_inp) + "_inp_data_in7am_red_mem.bin", "wb")

    data_in = pickle.load(fin)

    data7_in = []
    for i in range(365):
        for k in range(7, 10):
            if i <= 100:
                print(data_in[24 * i + k - time_inp][-4][0])
            data7_in.append(data_in[24 * i + k - time_inp])

    data7_inarr = np.array(data7_in)
    print(data7_inarr.shape)

    pickle.dump(data7_in, fin7)

    data_out = pickle.load(fout)
    data7_out = []

    for i in range(365):
        for k in range(7, 10):
            data7_out.append(data_out[24 * i + k - time_inp])

    data7_outarr = np.array(data7_out)
    print(data7_outarr.shape)

    pickle.dump(data7_out, fout7)

def create_7_to_9am_data():
    fout = open("../usefulData/" + str(time_inp) + "_inp_data_out_red.bin", "rb")
    fin = open("../usefulData/" + str(time_inp) + "_inp_data_in_red.bin", "rb")

    fout7 = open("../usefulData/" + str(time_inp) + "_inp_data_out729_red.bin", "wb")
    fin7 = open("../usefulData/" + str(time_inp) + "_inp_data_in729_red.bin", "wb")

    data_in = pickle.load(fin)

    data7_in = []
    for i in range(365):
        for k in range(7, 10):
            if i <= 1:
                print(data_in[24 * i + k - time_inp][-4][0])
            lst = data_in[24 * i + k - time_inp]
            lst2 = np.array(lst)
            lst2= lst2[:13:,:13:]
            print(lst2.shape)
            lst2 = lst2.flatten().tolist()
            for i in range(4):
                lst2.append(lst[-4+i][0])
            print(len(lst2))
            data7_in.append(lst2)

    data7_inarr = np.array(data7_in)
    print(data7_inarr.shape)

    pickle.dump(data7_in, fin7)

    data_out = pickle.load(fout)
    data7_out = []

    for i in range(365):
        for k in range(7, 10):
            data7_out.append(data_out[24 * i + k - time_inp])

    data7_outarr = np.array(data7_out)
    print(data7_outarr.shape)


    pickle.dump(data7_out, fout7)

def create_csv(bin_src):

    with open(bin_src,"rb") as f:
        data = pickle.load(f)

    path = bin_src[:-3] + "csv"
    print(path)
    lent = len(data)
    data = np.array(data).flatten()
    data = data.reshape((lent , int(len(data) / lent) ))
    pd.DataFrame(np.array(data)).to_csv(path)

def create_exp():
    fout = open("../usefulData/" + str(time_inp) + "_inp_data_out7am_red.bin", "rb")
    fin = open("../usefulData/" + str(time_inp) + "_inp_data_in7am_red.bin", "rb")

    fout7 = open("../usefulData/" + str(time_inp) + "_inp_data_out7amEXP_red.bin", "wb")
    fin7 = open("../usefulData/" + str(time_inp) + "_inp_data_in7amEXP_red.bin", "wb")

    data_in = pickle.load(fin)

    data7_in = []

    for data in data_in:
        dataEXP = np.array(data)
        dataEXP = np.exp(dataEXP)
        data7_in.append(dataEXP)

    data7_inarr = np.array(data7_in)
    print(data7_inarr.shape)
    print(data7_inarr[0])
    pickle.dump(data7_in, fin7)

    data_out = pickle.load(fout)
    data7_out = []

    for data in data_out:
        dataEXP = np.array(data)
        dataEXP = np.exp(dataEXP)
        data7_out.append(dataEXP)

    data7_outarr = np.array(data7_out)
    print(data7_outarr.shape)

    pickle.dump(data7_out, fout7)


create_total_input()
#create_7_to_9am_data()
create_5_to_10am_data()
#create_csv("../usefulData/" + str(time_inp) + "_inp_data_out729_red.bin")
#create_exp()
#create_166_input_new()