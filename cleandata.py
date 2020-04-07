import os
import pandas as pd

# clean up data
nBodies = 3
def appendHeader(inFile, outFile, fileNum):
    dataCols = ["finalFile", "eventID"]
    perParticleColumnsInput = ["m", "x", "y", "dx", "dy"]
    perParticleColumnsOutput = ["xf", "yf", "dxf", "dyf"]
    for col in perParticleColumnsInput:
        for i in range(nBodies):
            dataCols.append(col+str(i+1))
    dataCols.append("t")
    for col in perParticleColumnsOutput:
        for i in range(nBodies):
            dataCols.append(col+str(i+1))
    
    if "10_1" in inFile or "10_2" in inFile or "10_3" in inFile or "10_4" in inFile:
        df = pd.read_csv(inFile, index_col=False, names=dataCols)
    if "10_5" in inFile or "10_6" in inFile or "10_7" in inFile:
        df = pd.read_csv(inFile, index_col=False)

    with pd.option_context('mode.use_inf_as_null', True):
        df = df.dropna(axis=1)
        df = df.dropna(axis=0)
    df['finalFile'] = fileNum
    df.to_csv(dataDir+outFile, index=False)
dataDir = "/users/PAS1585/llavez99/data/nbody/3body/"
outFileName = "brutus10_"
for file in os.listdir(dataDir):
    print(file)
    fileNum = file[9:10]
    appendHeader(dataDir+file, outFileName+fileNum+"_3_final.csv", fileNum)