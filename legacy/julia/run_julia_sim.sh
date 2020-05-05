#PBS -N julia_sim
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=5000MB
#PBS -j oe
# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
module load julia/1.3.1	
julia threeBody.jl

