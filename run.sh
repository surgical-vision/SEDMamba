# For inference, run the following command:
python inference.py -exp Inference_SEDMamba -dp /path/to/your/SEDMamba/data -cls 1 -gpu_id cuda:0 -w 4 -weight /path/to/your/SEDMamba/weight/SEDMamba.pth

# For visualization, run the following command:
python visualization.py -lp /path/to/your/SEDMamba/exp_log/0.0001/Inference_SEDMamba -rp /path/to/your/SEDMamba/exp_log/0.0001/Inference_SEDMamba -sp /path/to/your/SEDMamba/exp_log/0.0001/Inference_SEDMamba

# For training, run the following command:
python train.py -exp SEDMamba -dp /path/to/your/SEDMamba/data -e 200 -l 1e-4 -cls 1 -gpu_id cuda:0 -w 4 -s 0