import os, sys 
from random import seed 
from random import SystemRandom

def grnum(minValue,maxValue):
    return SystemRandom().uniform(minValue, maxValue)

def create_data(batch, filenum, nEPerFile):
    seed(1)
    b = 10
    mb = 100

    for ifile in range(1,filenum+1):
        filename = "/nBodyData/inputs/indat_{0}_{1}.dat".format(batch, ifile)
        inp = open(filename, "w+")

        for iev in range(1,nEPerFile+1):
            # seed(random())
            # print(rvx1, rvx2, rvx3, rvy1, rvy2, rvy3)
            pxdata, pydata, mdata, vxdata, vydata = [], [], [], [], []
            for k in range(0,3):
                pxdata.append(grnum(-b,b))
                pydata.append(grnum(-b,b))
                # pzdata.append(grnum(-b,b))

                vxdata.append(grnum(-b/10,b/10))
                vydata.append(grnum(-b/10,b/10))
                # vzdata.append(grnum(-b/10,b/10))

                mdata.append(grnum(0.1, mb))

            # continue
            m = [mdata[0],mdata[1],mdata[2]]
            p = [[pxdata[0],pydata[0],0], [pxdata[1],pydata[1],0], [pxdata[2],pydata[2],0]]
            v = [[vxdata[0],vydata[0],0], [vxdata[1],vydata[1],0], [vxdata[2],vydata[2],0]]

            # create input file
            inp.write("{0},{1},{2},{3},{4},{5},{5},".format(m[0], p[0][0], p[0][1], p[0][2], v[0][0], v[0][1], v[0][2]))
            inp.write("{0},{1},{2},{3},{4},{5},{5},".format(m[1], p[1][0], p[1][1], p[1][2], v[1][0], v[1][1], v[1][2]))
            inp.write("{0},{1},{2},{3},{4},{5},{5}\n".format(m[2], p[2][0], p[2][1], p[2][2], v[2][0], v[2][1], v[2][2]))
        inp.close()
        print("file created: {0}".format(filename))


if __name__ == "__main__":
    # configurable sim parameters 
    batch = 6
    nFiles = 1
    nEventsPerFile = 50
    timeStampsPerEvent = 2560 
    tEnd = 10
    pMax = 4

    create_data(batch, nFiles, nEventsPerFile)