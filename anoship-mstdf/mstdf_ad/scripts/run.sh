python main.py --dataset PSM --anomaly_ratio 1
python main.py --dataset SMD --p 2 --anomaly_ratio 0.5
python main.py --dataset SWaT  --p 3 --hidden_dim 12 --anomaly_ratio 1
python main.py --dataset MSL --anomaly_ratio 1
python main.py --dataset SMAP  --anomaly_ratio 1

python main.py --dataset NSMC-EV-3 --level 4 --itr 1 --window_size 128 --step_size 32 --kernel_size 8 --stride 4
python main.py --dataset NSMC-EV-4 --level 4 --itr 1 --window_size 128 --step_size 32
