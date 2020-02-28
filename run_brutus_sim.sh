#PBS -N brutus_sim
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=10
#PBS -l mem=10GB
#PBS -j oe
# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
cd /users/PAS1585/llavez99/work/nbody/NvMgeneration/
./main.exe 1

