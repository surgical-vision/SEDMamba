import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import argparse

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_label_data(path):
    return [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if '_label.csv' in file]

def plot_scores(label_data, score_ours, save_path, video_id):
    fig, ax_score = plt.subplots(1, 1, figsize=(30, 5))
    cmap_score = ListedColormap(['white', 'red'])
    ax_score.imshow(label_data.iloc[:, 0].values[np.newaxis], aspect='auto', cmap=cmap_score, alpha=0.3, extent=[0, len(label_data), 0, 1])
    
    ax_score.plot(score_ours, color='green', label='Ours')
    
    ax_score.set_ylabel('Error Probability', fontsize=20)
    ax_score.set_xlabel('Frame Index', fontsize=20)
    ax_score.legend(fontsize=15, framealpha=0.3, loc='upper left')
    ax_score.set_xticks(np.arange(0, len(label_data), 100))
    ax_score.set_yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/video_{video_id}_score.png", dpi=600)

def main(label_data_path, result_path, saved_path):
    create_directory(saved_path)
    label_datas = load_label_data(label_data_path)
    for label_data_path in label_datas:
        label_data = pd.read_csv(label_data_path)
        video_id = label_data_path.split('_')[-2]
        score_ours = pd.read_csv(f"{result_path}/video_{video_id}_score.csv")
        plot_scores(label_data, score_ours, saved_path, video_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scores from label and prediction data.')
    parser.add_argument('-lp', '--label_data_path', type=str, required=True, help='Path to the ground truth labels.')
    parser.add_argument('-rp', '--result_path', type=str, required=True, help='Path to the prediction results.')
    parser.add_argument('-sp', '--saved_path', type=str, required=True, help='Path to save the plots.')
    
    args = parser.parse_args()
    main(args.label_data_path, args.result_path, args.saved_path)