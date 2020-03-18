import os, sys 
from random import seed 
from random import SystemRandom
from random import randint

def grnum(minValue,maxValue):
    return SystemRandom().uniform(minValue, maxValue)

def create_data(batch, filenum, nEPerFile, nBodies):
    seed(0)

    n = nBodies

    pb = 10    # bounds for position
    vb = 1     # bounds for velocity
    mb = 100   # bounds for mass

    for ifile in range(1,filenum+1):

        filename = "/Users/brandonmanley/Documents/nBody/data/inputs/indat_{0}_{1}_{2}.dat".format(batch, ifile, n)
        inp = open(filename, "w+")

        for iev in range(1,nEPerFile+1):

            pxdata, pydata, pzdata, mdata, vxdata, vydata, vzdata = [], [], [], [], [], [], []
          
            for k in range(0, n):
                pxdata.append(grnum(-pb,pb))
                pydata.append(grnum(-pb,pb))
                pzdata.append(0)
                # pzdata.append(grnum(-pb,pb))

                vxdata.append(grnum(-vb, vb))
                vydata.append(grnum(-vb, vb))
                vzdata.append(0)
                # vzdata.append(grnum(-vb, vb))

                mdata.append(grnum(1, mb))

            # create input file
            for k in range(0, n):
                if k != n-1:
                    inp.write("{0},{1},{2},{3},{4},{5},{6},".format(mdata[k], pxdata[k], pydata[k], pzdata[k], vxdata[k], vydata[k], vzdata[k]))
                else:
                    inp.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(mdata[k], pxdata[k], pydata[k], pzdata[k], vxdata[k], vydata[k], vzdata[k], n))
                    
        inp.close()
        print("files created: {0}".format(filename))


if __name__ == "__main__":
	
    nBodies = -1 
    if len(sys.argv) < 2: 
        print("Usage: createInData.py # of bodies")
        exit()
    else:
        try:
            nBodies = int(sys.argv[1])
        except: 
            print("Error: Enter positive integer")
            exit()
        
    # configurable data parameters
    batch = 10
    nFiles = 1
    nEventsPerFile = 100
    create_data(batch, nFiles, nEventsPerFile, nBodies)