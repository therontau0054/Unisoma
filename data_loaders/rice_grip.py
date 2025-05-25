import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Rice_Grip_Rollout(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        frames_num: int = 41,
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.frames_num = frames_num
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"rice_grip_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        current_deformable_1 = np.stack(
            (
                [
                    f[index]["rice_positions"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        rigid_1 = np.stack(
            (
                [
                    f[index]["gripper_1_positions"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )
        load_1 = np.stack(
            (
                [
                    f[index]["gripper_1_positions"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        load_1 = load_1 - rigid_1
        load_1 = np.concatenate(
            (rigid_1, load_1),
            axis = -1
        )

        rigid_2 = np.stack(
            (
                [
                    f[index]["gripper_2_positions"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )
        load_2 = np.stack(
            (
                [
                    f[index]["gripper_2_positions"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        load_2 = load_2 - rigid_2
        load_2 = np.concatenate(
            (rigid_2, load_2),
            axis = -1
        )
        next_deformable_1 = np.stack(
            (
                [
                    f[index]["rice_positions"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        target_deformable_1 = next_deformable_1 - current_deformable_1
            
        return [
                   current_deformable_1.astype(self.numeric_type)
               ], \
               [
                   rigid_1.astype(self.numeric_type),
                   rigid_2.astype(self.numeric_type)
               ], \
               [
                   load_1.astype(self.numeric_type),
                   load_2.astype(self.numeric_type)
               ], \
               [
                   next_deformable_1.astype(self.numeric_type)
               ], \
               [
                   target_deformable_1.astype(self.numeric_type)
               ]   

    def __len__(self):
        return len(self.f)
    