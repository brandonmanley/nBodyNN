import os, sys 
from random import seed 
from random import SystemRandom
from random import randint
import numpy as np
from keras import models
from keras import layers

workDir = '/mnt/c/users/llave/Documents/nBody'
fileDir = '/mnt/c/users/llave/Documents/nBody'

def grnum(minValue,maxValue):
    return SystemRandom().uniform(minValue, maxValue)

def create_data(batch, filenum, nEPerFile,epsilon):
    seed(1)

    pb = 10    # bounds for position
    vb = 1     # bounds for velocity
    mb = 100   # bounds for mass
    nb = 10    # upper bound for # of pls

    for ifile in range(1,filenum+1):

        filename = fileDir + "/indat_{0}_{1}.dat".format(batch, ifile)
        inp = open(filename, "w+")

        iev = 0
        #for iev in range(1,nEPerFile+1):
        while iev < nEPerFile+1:

            # n = randint(2,nb)
            n = 3 
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

            testArray = np.concatenate([mdata,pxdata,pydata,vxdata,vydata])
            div = testDiv(testArray,epsilon)
            if(div): 
            	continue

            # create input file
            for k in range(0, n):
                if k != n-1:
                    inp.write("{0},{1},{2},{3},{4},{5},{6},".format(mdata[k], pxdata[k], pydata[k], pzdata[k], vxdata[k], vydata[k], vzdata[k]))
                else:
                    inp.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(mdata[k], pxdata[k], pydata[k], pzdata[k], vxdata[k], vydata[k], vzdata[k], n))
            
            iev += 1
            if(iev%100==0): print(iev)

        inp.close()
        print("files created: {0}".format(filename))

def testDiv(testArray,epsilon):

	testArray = testArray.astype('float64')

	activation = 'tanh'
	optimizer = 'adam'
	batch_size = 20      
	network = models.Sequential()
	network.add(layers.Dense(64,activation=activation,input_shape=(15,)))
	network.add(layers.Dense(2,activation='softmax'))
	network.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
	network.load_weights(workDir + '/weights/div_weights.h5')

	pred = network.predict(np.array([testArray]))
	if(pred[0][1] > epsilon): div = True
	else: div = False

	return div

if __name__ == "__main__":
    # configurable data parameters 
    batch = 0
    nFiles = 1
    nEventsPerFile = 1000
    epsilon = 0.1
    create_data(batch, nFiles, nEventsPerFile,epsilon)