# export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
# export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export PYTHONPATH=/homes/55/junlin/MMaDA:$PYTHONPATH
accelerate launch --config_file accelerate_configs/test_run.yaml --main_process_port=8888 training/train_t2m.py config=configs/t2m_instruct.yaml