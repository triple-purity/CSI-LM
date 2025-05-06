# LLM-Distillation
python llm_distillation.py --data_path /home/share/data/CSI-Data-Complex/env1 \--output_dir /home/share/data/csi_model_param \--time_length 800 \--data_norm_type mean_std \--data_key csi_data \--input_dim 90 \--reprogramming True \--epoch 100 \--lr 5e-5

python llm_distillation.py --data_path /home/share/data/CSI-Ratio/env1 \--output_dir /home/share/data/csi_ratio_model_param \--antenna_num 2 \--time_length 600 \--extract_method csi-ratio \--data_norm_type mean_std \--data_key csi-ratio \--d_model 512 \--input_dim 30 \--reprogramming True \--epoch 100 \--lr 3e-5

python llm_distillation.py --data_path /home/share/data/DFS/env1 \--output_dir /home/share/data/dfs_model_param \--antenna_num 3 \--time_length 1800 \--extract_method dfs \--data_norm_type mean_std \--data_key dfs \--input_dim 121 \--reprogramming True \--epoch 100 \--lr 3e-5


# Fintue-LLM
python llm_fintue.py --data_path /home/share/data/CSI-Data-Complex/env1 \--output_dir /home/share/data/csi_model_param \--time_length 600 \--data_norm_type mean_std \--data_key csi_data \--input_dim 90 \--reprogramming True \--epoch 100 \--lr 5e-5

python llm_fintue.py --data_path /home/share/data/CSI-Ratio/env1 \--output_dir /home/share/data/csi_ratio_model_param \--antenna_num 2 \--time_length 600 \--extract_method csi-ratio \--data_norm_type mean_std \--data_key csi-ratio \--input_dim 60 \--reprogramming True \--epoch 100 \--lr 5e-5

python llm_fintue.py --data_path /home/share/data/DFS/env1 \--output_dir /home/share/data/dfs_model_param \--antenna_num 3 \--time_length 1800 \--extract_method dfs \--data_norm_type mean_std \--data_key dfs \--input_dim 121 \--reprogramming True \--epoch 100 \--lr 5e-5


# LLM-GAN


python train_llm_gan.py --data_path /home/share/data/DFS/env1 \--output_dir /home/share/data/dfs_model_param \--antenna_num 3 \--time_length 1800 \--extract_method dfs \--data_norm_type mean_std \--data_key dfs \--input_dim 121 \--reprogramming True \--epoch 400