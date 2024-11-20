model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

batch_size=8
d_model=32
d_ff=128

seq_len=96

comment='TimeLLM-Weather'

for pred_len in 24 36 48 96 192
do

model_id=weather_${seq_len}_${pred_len}

accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 --dynamo_backend no run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  2>&1 | tee -a logs/$model_id.log

done
