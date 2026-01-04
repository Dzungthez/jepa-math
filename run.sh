run_regular() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  seed=${4}
  dataset=${5}

  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} s=${seed} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 train.py \
    --train_file datasets/${dataset}_train.jsonl \
    --eval_file datasets/${dataset}_test.jsonl \
    --output_dir=./checkpoints --num_epochs=${epoch} --seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate}
  python evaluate.py --model_name=./checkpoints \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl \
    --original_model_name=${base_model_name} | tee -a output.txt
}

run_jepa() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=8 train.py \
    --train_file datasets/${dataset}_train.jsonl \
    --eval_file datasets/${dataset}_test.jsonl \
    --output_dir=./checkpoints --num_epochs=${epoch} --seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --additive_mask=True --model_name=${base_model_name} --learning_rate=${learning_rate}
  python evaluate.py --model_name=./checkpoints \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl \
    --original_model_name=${base_model_name} | tee -a output.txt
}

# Model configurations
models=(meta-llama/Llama-3.2-1B-Instruct apple/OpenELM-1_1B-Instruct google/gemma-2-2b-it \
        microsoft/phi-1_5 allenai/OLMo-2-0425-1B-Instruct)
non_it_models=(meta-llama/Llama-3.2-1B apple/OpenELM-1_1B google/gemma-2-2b \
               microsoft/phi-1_5 allenai/OLMo-2-0425-1B)
datasets=(gsm8k spider synth turk)

# Example training runs
for seed in 82 23 37 84 4
do
  model_name=meta-llama/Llama-3.2-1B-Instruct
  learning_rate=2e-5
  dataset=gsm8k
  for lbd in 0.1 0.5 1.0
  do
    for predictors in 1 2 3
    do
      run_jepa ${model_name} ${learning_rate} 3 -1 ${predictors} ${seed} ${lbd} ${dataset}
    done
  done
done

for seed in 82 23 37 84 4
do
  model_name=meta-llama/Llama-3.2-1B-Instruct
  learning_rate=2e-5
  dataset=spider
  for lbd in 0.1 0.5 1.0
  do
    for predictors in 1 2 3
    do
      run_jepa ${model_name} ${learning_rate} 3 -1 ${predictors} ${seed} ${lbd} ${dataset}
    done
  done
done

