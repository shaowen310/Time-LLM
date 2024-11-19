model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

batch_size=16
d_model=32
d_ff=128

seq_len=96

comment='TimeLLM-Exchange'

for pred_len in 24 36 48 96 192
do

model_id=exchange_${seq_len}_${pred_len}

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id $model_id \
  --model $model_name \
  --data Exchange \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 16 \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  2>&1 | tee -a logs/$model_id.log
done
