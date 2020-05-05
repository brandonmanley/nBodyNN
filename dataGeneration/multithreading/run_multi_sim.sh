#PBS -N brutus_multithreaded_sim
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=40
#PBS -j oe

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
cd /users/PAS1585/llavez99/work/nbody/NvMgeneration/
export OMP_NUM_THREADS=40
./main.exe 1 3
