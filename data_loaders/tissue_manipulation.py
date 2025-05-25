import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class DataLoader_Tissue_Manipulation_Rollout(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        frames_num: int = 105,
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.frames_num = frames_num
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"tissue_manipulation_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        current_deformable_1 = np.stack(
            (
                [
                    f[index]["tissue_mesh_positions"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )
        rigid_1 = np.stack(
            (
                [
                    f[index]["gripper_position"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )
        load_1 = np.stack(
            (
                [
                    f[index]["gripper_position"][k]
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

        rigid_2 = f[index]["grasping_position"].reshape(1, 1, -1)
        rigid_2 = rigid_2.repeat(self.frames_num - 1, axis = 0)
        load_2 = np.concatenate(
            (rigid_2, np.zeros_like(rigid_2)),
            axis = -1
        )

        next_deformable_1 = np.stack(
            (
                [
                    f[index]["tissue_mesh_positions"][k]
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
    
    

class DataLoader_Tissue_Manipulation_LongTime(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        target_frame: int = -1,
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.target_frame = target_frame
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"tissue_manipulation_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        init_deformable_1 = np.array(f[index]["tissue_mesh_positions"][0])
        final_deformable_1 = np.array(f[index]["tissue_mesh_positions"][self.target_frame])
        target_deformable_1 = final_deformable_1 - init_deformable_1

        rigid_1 = np.array(f[index]["gripper_position"][0])
        load_1 = np.array(f[index]["gripper_position"][self.target_frame]) - rigid_1
        load_1 = np.concatenate(
            (rigid_1, load_1),
            axis = -1
        )

        rigid_2 = np.array(f[index]["grasping_position"]).reshape(1, -1)
        load_2 = np.concatenate(
            (rigid_2, np.zeros_like(rigid_2)),
            axis = -1
        )
        
            
        return [init_deformable_1.astype(self.numeric_type)], \
               [
                   rigid_1.astype(self.numeric_type),
                   rigid_2.astype(self.numeric_type)
               ], \
               [
                   load_1.astype(self.numeric_type),
                   load_2.astype(self.numeric_type)
               ], \
               [
                   final_deformable_1.astype(self.numeric_type)
               ], \
               [
                   target_deformable_1.astype(self.numeric_type)
               ]    

    def __len__(self):
        return len(self.f)
    