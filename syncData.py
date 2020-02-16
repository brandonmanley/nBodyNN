import os 

localPath = "/nBodyData/"
drivePath = "/Users/brandonmanley/Desktop/OneDrive - The Ohio State University/nBody Data/"
drivePath2 = "/Users/brandonmanley/Desktop/OneDrive\ -\ The\ Ohio\ State\ University/nBody\ Data/"
dirs = ["mathSim/", "mathMeta/", "brutusSim/"]

for direc in ["mathSim/", "julSim/"]:
    for file in os.listdir(localPath+direc):
        if "batch" not in file: continue
        if ".gz" not in file:
            os.system("gzip {0}".format(localPath+direc+file))
    
    for file in os.listdir(drivePath+direc):
        if file not in os.listdir(localPath+direc): 
            os.system("rm {0}".format(drivePath2+direc+file))
    os.system("cp -r {0}*.gz {1}".format(localPath+direc, drivePath2+direc))

for file in os.listdir(localPath+"inputs/"):
    if ".gz" not in file:
        os.system("gzip {0}".format(localPath+"inputs/"+file))

    for file in os.listdir(drivePath+"inputs/"):
        if file not in os.listdir(localPath+"inputs/"): 
            os.system("rm {0}".format(drivePath2+"inputs/"+file))
    os.system("cp -r {0}*.gz {1}".format(localPath+"inputs/", drivePath2+"inputs/"))
