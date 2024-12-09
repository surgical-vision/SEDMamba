import os
from torch.utils.data import Dataset
import pickle
import numpy as np
import re
from typing import Optional, Tuple, List


def extract_number(file_name: str) -> int:
    match = re.search(r"(\d+)", file_name)
    return int(match.group()) if match else 0


class CustomVideoDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = sorted(os.listdir(root_dir), key=extract_number)

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray, str]:
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name)
        with open(video_path, "rb") as f:
            video_data = pickle.load(f)
        features = video_data["feature"].astype("float32")
        e_labels = video_data["error_GT"]
        video_length = len(e_labels)

        if self.transform:
            features = self.transform(features)

        # Return the frames of the video as a list and its corresponding label
        return features, video_length, e_labels, video_name
