x = [1,2,3,4,5,6,7,8,9,10]
k = 3
fold = 0
for i in range(len(x)):
    if i% int(len(k)/2) == 0:
        fold += 1