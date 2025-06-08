export PYTHONPATH=/homes/55/junlin/MMaDA:$PYTHONPATH
# accelerate launch --config_file accelerate_configs/test_run.yaml --main_process_port=8888 training/train_t2m.py config=configs/t2m_instruct.yaml

accelerate launch --config_file accelerate_configs/test_run.yaml --main_process_port=8888 training/train_t2m.py config=configs/t2m_test.yaml
