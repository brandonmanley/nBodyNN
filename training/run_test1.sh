#PBS -N run_test1
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=40
#PBS -l mem=5GB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2 
python /users/PAS1585/llavez99/work/nbody/nBodyNN/training/test.py
