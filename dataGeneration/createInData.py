import os, sys 
from random import seed 
from random import SystemRandom
from random import randint
import numpy as np

fileDir = '/users/PAS1585/llavez99/data/nbody/'

def grnum(minValue,maxValue):
    return SystemRandom().uniform(minValue, maxValue)

def create_data(batch, nBodies, file_i, file_f, nEPerFile):
    seed(1)

    pb = 10    # bounds for position
    vb = 1     # bounds for velocity
    mb = 100   # bounds for mass
    nb = 10    # upper bound for # of pls

    for ifile in range(file_i,file_f+1):

        filename = fileDir + str(nBodies) + "body/indat_{0}_{1}.dat".format(batch, ifile)
        inp = open(filename, "w+")

        
        for iev in range(1,nEPerFile+1):

            # n = randint(2,nb)
            n = nBodies 
            pxdata, pydata, pzdata, mdata, vxdata, vydata, vzdata = [], [], [], [], [], [], []

            for k in range(0,n):
                pxdata.append(grnum(-pb,pb))
                pydata.append(grnum(-pb,pb))
                pzdata.append(0)
                # pzdata.append(grnum(-pb,pb))

                vxdata.append(grnum(-vb, vb))
                vydata.append(grnum(-vb, vb))
                vzdata.append(0)
                # vzdata.append(grnum(-vb, vb))

                mdata.append(grnum(0.001, mb))

            # create input file
            for k in range(0, n):
                if k != n-1:
                    inp.write("{0},{1},{2},{3},{4},{5},{6},".format(mdata[k], pxdata[k], pydata[k], pzdata[k], vxdata[k], vydata[k], vzdata[k]))
                else:
                    inp.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(mdata[k], pxdata[k], pydata[k], pzdata[k], vxdata[k], vydata[k], vzdata[k], n))

        inp.close()
        print("files created: {0}".format(filename))

if __name__ == "__main__":
    # configurable data parameters 
    batch = 10
    nBodies = 3
    file_i = 8
    file_f = 10
    nEventsPerFile = 1000
    create_data(batch, nBodies, file_i, file_f, nEventsPerFile)
