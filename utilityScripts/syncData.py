import os 
import progressbar

dirList = ["inputs/", "julSim/", "mathSim/", "brutusSim/"]
dataPath = "/nBodyData/"
oneDrivePath = "/Users/brandonmanley/Desktop/OneDrive\ -\ The\ Ohio\ State\ University/nBody\ Data/"

# remove all files from compressed directories 
for dir in dirList:
    os.system("rm -r {0}*.gz".format(dataPath+"compressed/"+dir))

# gzip all files to compressed directories
for dir in dirList:
    for file in progressbar.progressbar(os.listdir(dataPath+dir)):
        filePath = dataPath+dir+file
        compFilePath = dataPath+"/compressed/"+dir+file 

        os.system("gzip -c {0} > {1}.gz".format(filePath, compFilePath))

# sync oneDrive w/ compressed dirs
for dir in dirList:
    drivePath = oneDrivePath+dir

    for file in os.listdir(dataPath+"compressed/"+dir):
        if "DS" in file: continue

        filePath = dataPath+"compressed/"+dir+file 
        
        os.system("cp {0} {1}".format(filePath, drivePath))
        
