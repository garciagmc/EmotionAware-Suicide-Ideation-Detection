#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate bert2
#bert-serving-start -model_dir pretrainedModels/uncased_L-12_H-768_A-12/ -num_worker=10 -max_seq_len=40
bert-serving-start -model_dir pretrainedModels/cased_L-12_H-768_A-12/ -num_worker=10
conda deactivate