
###random
#### ignore will soon be removed

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

sorted_x = sorted(bij.items(), key=lambda kv: kv[1])

print(sorted_x)
new_bij = dict([])
count = 0

for st,val in sorted_x:
    if val!= -1:
        new_bij[st] = count
        count+=1

print(new_bij)