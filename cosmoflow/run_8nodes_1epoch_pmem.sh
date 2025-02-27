#!/bin/bash
#PBS -q gen_S
#PBS -A INSITU
#PBS -T openmpi
#PBS -l elapstim_req=00:20:00
#PBS -v NQSV_MPI_VER=nvhpc-hpcx-cuda11/24.5
#PBS -b 8

cd ${PBS_O_WORKDIR}

module load cuda/11.8.0
module load cudnn/8.9.7/cuda11
module load openmpi/$NQSV_MPI_VER

source .venv/bin/activate

mpirun ${NQSV_MPIOPTS} -np 8 -npernode 1 --bind-to none -x PATH -x LD_LIBRARY_PATH \
python3 train.py -d --rank-gpu --mlperf --n-epochs 1 \
    --data-dir /work/INSITU/share/cosmoUniverse_2019_05_4parE_tf_v2 \
    --stage-dir /pmem \
    --amp
