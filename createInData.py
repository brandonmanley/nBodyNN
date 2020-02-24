import os, sys 
from random import seed 
from random import random 

def create_data(batch, filenum, nEPerFile):
    seed(1)
    mins = [[-5,3], [-1,-7], [7,3]]
    maxs = [[-2.5,5], [2,-2], [20,15]]
    for ifile in range(1,filenum+1):
        filename = "/users/PAS1585/llavez99/work/nbody/indat_{0}_{1}.dat".format(batch, ifile)
        inp = open(filename, "w+")

        for iev in range(1,nEPerFile+1):
            # seed(random())
            rvx1, rvx2, rvx3 = mins[0][0]+(random()*(maxs[0][0] - mins[0][0])), mins[1][0]+(random()*(maxs[1][0] - mins[1][0])), mins[2][0]+(random()*(maxs[2][0] - mins[2][0]))
            rvy1, rvy2, rvy3 = mins[0][1]+(random()*(maxs[0][1] - mins[0][1])), mins[1][1]+(random()*(maxs[1][1] - mins[1][1])), mins[2][1]+(random()*(maxs[2][1] - mins[2][1]))
            # print(rvx1, rvx2, rvx3, rvy1, rvy2, rvy3)
            
            # continue
            m = [150,120,130]
            p = [[rvx1,rvy1,0], [rvx2,rvy2,0], [rvx3,rvy3,0]]
            v = [[0,0,0], [0,0,0], [0,0,0]]

            # create input file
            inp.write("{0},{1},{2},{3},0,0,0,".format(m[0], p[0][0], p[0][1], p[0][2]))
            inp.write("{0},{1},{2},{3},0,0,0,".format(m[1], p[1][0], p[1][1], p[1][2]))
            inp.write("{0},{1},{2},{3},0,0,0\n".format(m[2], p[2][0], p[2][1], p[2][2]))
        inp.close()
        print("file created: {0}".format(filename))


if __name__ == "__main__":
    # configurable sim parameters 
    batch = 4
    nFiles = 1
    nEventsPerFile = 10000
    timeStampsPerEvent = 2560 
    tEnd = 10
    pMax = 4

    create_data(batch, nFiles, nEventsPerFile)

    

    
''' 
FIXME: 
This script:
- add config file
- add drive sync command

Mathematica: 
- Read config file

Brutus:
- Read config file
''' 

    # os.system("make main.exe")
    # os.system("./main.exe")