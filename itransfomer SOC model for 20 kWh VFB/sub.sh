export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/SOC/ \
  --data_path Dataset_discharge.csv \
  --data SOC \
  --model_id charge_96_96 \
  --model $model_name \
  --features MS \
  --target Ture_SOC \
  --freq s \
  --start_cycle 0 \
  --end_cycle 25 \
  --test_cycle 26 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1 \


##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path /home/DNL17_DUT/Pytorch/dataset/weather/ \
##  --data_path weather.csv \
##  --model_id weather_96_192 \
##  --model $model_name \
##  --data custom \
##  --features M \
##  --seq_len 96 \
##  --label_len 48 \
##  --pred_len 192 \
##  --e_layers 3 \
##  --d_layers 1 \
##  --factor 3 \
##  --enc_in 10 \
##  --dec_in 10 \
##  --c_out 10 \
##  --des 'Exp' \
##  --d_model 512\
##  --d_ff 512\
##  --itr 1 \
##
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path /home/DNL17_DUT/Pytorch/dataset/weather/ \
##  --data_path weather.csv \
##  --model_id weather_96_336 \
##  --model $model_name \
##  --data custom \
##  --features M \
##  --seq_len 96 \
##  --label_len 48 \
##  --pred_len 336 \
##  --e_layers 3 \
##  --d_layers 1 \
##  --factor 3 \
##  --enc_in 10 \
##  --dec_in 10 \
##  --c_out 10 \
##  --des 'Exp' \
##  --d_model 512\
##  --d_ff 512\
##  --itr 1 \
##
##
##python -u run.py \
##  --task_name long_term_forecast \
##  --is_training 1 \
##  --root_path /home/DNL17_DUT/Pytorch/dataset/weather/ \
##  --data_path weather.csv \
##  --model_id weather_96_720 \
##  --model $model_name \
##  --data custom \
##  --features M \
##  --seq_len 96 \
##  --label_len 48 \
##  --pred_len 720 \
##  --e_layers 3 \
##  --d_layers 1 \
##  --factor 3 \
##  --enc_in 10 \
##  --dec_in 10 \
##  --c_out 10 \
##  --des 'Exp' \
##  --d_model 512\
##  --d_ff 512\
##  --itr 1
