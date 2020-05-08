#PBS -N train_3body_distributed
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=5:gpus=2
#PBS -l mem=1GB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2 
python /users/PAS1585/llavez99/work/nbody/nBodyNN/training/distributed_training.py
