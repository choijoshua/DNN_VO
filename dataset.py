import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import v2
import pandas as pd
from PIL import Image
import numpy as np
from utils import get_relative_pose


class VODataset(Dataset):
    def __init__(self, config, image_dir, pose_path, transform=None):
        # Initialize data, download, etc.
        self.image_dir = Path(image_dir)

        if transform is None:
            self.transforms = v2.Compose(
                [
                    v2.Resize(config.height, config.width),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transforms = transform

        self.clip_size = config.size
        self.clips = []
        for i in range(len(self.image_paths) - config.clip_size + 1):
            self.clips.append((i, i + config.clip_size))

        df = pd.read_csv(pose_path)

        self.poses = {}

        for _, row in df.iterrows():
            image_key = row["image"]

            pose_params = {
                "position": np.array([row["x"], row["y"], row["z"]]),
                "orientation": np.array([row["roll"], row["pitch"], row["yaw"]]),
            }

            self.poses[image_key] = pose_params

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_indices = self.clips[idx]

        # Load images
        images = []
        for img_idx in clip_indices:
            img_path = self.image_dir / f"{img_idx}.jpg"
            img = Image.open(img_path).convert("RGB")
            img = self.transforms(img)
            images.append(img)

        # Calculate relative poses
        poses = []
        for i in range(len(clip_indices) - 1):
            curr_idx = clip_indices[i]
            next_idx = clip_indices[i + 1]

            curr_pose = self.poses[curr_idx]
            next_pose = self.poses[next_idx]

            rel_pose = get_relative_pose(curr_pose, next_pose)
            poses.append(rel_pose)

        # Stack tensors
        image_tensor = torch.stack(images, dim=0)
        pose_tensor = torch.tensor(np.array(poses), dtype=torch.float32)

        return image_tensor, pose_tensor
