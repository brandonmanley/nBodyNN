#PBS -N train_5body
#PBS -l walltime=48:00:00
#PBS -l nodes=1:ppn=10
#PBS -l mem=5000MB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2 
python /users/PAS1585/llavez99/work/nbody/nBodyNN/nn_train_test5body.py 
