import os 
import progressbar

dirList = ["inputs/", "julSim/", "mathSim/", "brutusSim/"]
dataPath = "/nBodyData/"
oneDrivePath = "/Users/brandonmanley/Desktop/oneDrive/nBodyData/"

# remove all files from compressed directories 
print("Removing compressed files")
for dir in dirList:
    os.system("rm -r {0}*.gz".format(dataPath+"compressed/"+dir))

# gzip all files to compressed directories
print("Zipping uncompressed files...")
for dir in dirList:
    for file in progressbar.progressbar(os.listdir(dataPath+dir)):
        filePath = dataPath+dir+file
        compFilePath = dataPath+"/compressed/"+dir+file 
        os.system("gzip -c {0} > {1}.gz".format(filePath, compFilePath))

# sync oneDrive w/ compressed dirs
print("Syncing oneDrive folder...")
for dir in dirList:
    drivePath = oneDrivePath+dir
    for file in os.listdir(dataPath+"compressed/"+dir):
        if "DS" in file: continue
        filePath = dataPath+"compressed/"+dir+file 
        os.system("cp {0} {1}".format(filePath, drivePath))