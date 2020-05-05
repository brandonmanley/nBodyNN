import os

file_i = 8
file_f = 10
nBodies = 3
for i in range(file_i, file_f+1):
    print("Submitting indata file",i,"running on",nBodies)
    os.system("qsub -A PAS1585 run_brutus_sim.sh {0} {1}".format(i, nBodies))