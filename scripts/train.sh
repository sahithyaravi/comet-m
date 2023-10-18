
# Please specify --data_dir that contains [train|val|test].[source|target].
# Please specify --model_name_or_path as facebook/bart-large or a path to a saved COMET checkpoint e.g "comet-atomic_2020_BART"
# Finetune COMET
echo "Training COMET"
# train_path="training_data/mei_datasets/MEI"
train_path="/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/training_data/multi-event/data-m-v9"
output_path="comet-m-v1"
generated_file="${output_path}/test_generations.txt"
epochs=2

CUDA_VISIBLE_DEVICES=3 python -u comet_atomic2020_bart/finetune.py\
    --task summarization \
    --num_workers 4 \
    --learning_rate=1e-5 \
    --gpus 1 \
    --do_train \
    --n_val -1 \
    --val_check_interval 1.0 \
    --data_dir $train_path\
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --warmup_steps=400 \
    --output_dir $output_path\
    --num_train_epochs $epochs \
    --model_name_or_path facebook/bart-large\
    --atomic \



source="${train_path}/test.source"
tgt="${train_path}/test.target"
model="${output_path}/best_tfmr"

python -u system_eval/automatic_eval_multiple.py --comet_model $model --ground_truth_target_file $tgt --ground_truth_source_file $source --save_path $output_path