source r0.3.2
export PATH=~/.local/bin/:$PATH
export LC_ALL=en_US.UTF-8
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=32 --comment=spring-submit \
python3 -m torch.distributed.launch --nproc_per_node=8 train_from_scratch.py \
                            --train_dir /mnt/lustre/lichuming/0chen/Image-pytorch/train --test_dir /mnt/lustre/lichuming/0chen/Image-pytorch/val
