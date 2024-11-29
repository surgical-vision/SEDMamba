# For inference, run the following command:
python inference.py -exp Inference_SEDMamba -dp /media/HDD1/jialang/error_detection_morty/SEDMamba/data -cls 1 -gpu_id cuda:0 -w 4 -weight /media/HDD1/jialang/error_detection_morty/SEDMamba/weight/SEDMamba.pth

# For visualization, run the following command:
python visualization.py -lp /media/HDD1/jialang/error_detection_morty/SEDMamba/exp_log/0.0001/Inference_SEDMamba -rp /media/HDD1/jialang/error_detection_morty/SEDMamba/exp_log/0.0001/Inference_SEDMamba -sp /media/HDD1/jialang/error_detection_morty/SEDMamba/exp_log/0.0001/Inference_SEDMamba

# For training, run the following command:
python train.py -exp SEDMamba -dp /media/HDD1/jialang/error_detection_morty/SEDMamba/data -e 200 -l 1e-4 -cls 1 -gpu_id cuda:0 -w 4 -s 0