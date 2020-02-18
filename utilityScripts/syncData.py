import os 

dirList = ["inputs/"]
dataPath = "/nBodyData/"
oneDrivePath = "/Users/brandonmanley/Desktop/OneDrive\ -\ The\ Ohio\ State\ University/nBody\ Data/"


# gzip all files to compressed directories
for dir in dirList:
    for file in os.listdir(dataPath+dir):
        filePath = dataPath+dir+file
        compFilePath = dataPath+"/compressed/"+dir+file 

        os.system("gzip -c {0} > {1}.gz".format(filePath, compFilePath))
        
