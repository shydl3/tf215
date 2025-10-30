# FIRST：
conda env create -f environment_full.yml

用 conda 从 NVIDIA channel 安装 cuda-nvcc 11.8（带 nvvm/libdevice）
conda install -c nvidia cuda-nvcc=11.8 -y

## 设置nvcc环境变量，顺利编译
echo $CONDA_PREFIX
/home/ldonglin@id.sdsu.edu/miniconda3/envs/test1


find $CONDA_PREFIX -type f -path "*/nvvm/libdevice/libdevice.10.bc"
/home/ldonglin@id.sdsu.edu/miniconda3/envs/test1/nvvm/libdevice/libdevice.10.bc


export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_DISABLE_XLA=1
export TF_ENABLE_ONEDNN_OPTS=0

## 后台挂起训练
screen -S train

screen -ls

CTRL A + D: detatch screen session

RECONNECT: screen -r train

TERMINATE SSSCREEN: IN SESSION: exit

   #### Configuration
   Put the data set into the `data/github` directory under `keras`
   
   Edit hyper-parameters and settings in `config.py`
   
   #### Train and Evaluate
   
   ```bash
   python main.py --mode train
   python main.py --mode eval
