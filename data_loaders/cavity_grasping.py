import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Cavity_Grasping_Rollout(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        dt: int = 1,
        frames_num: int = 105,
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.dt = dt
        self.frames_num = frames_num
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"cavity_grasping_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        dt = self.dt
        f = self.f
        current_deformable_1 = np.stack(
            (
                [
                    f[index]["tissue_mesh_positions"][k]
                    for k in range(0, self.frames_num - 1, dt)
                ]
            ),
            axis = 0
        )

        rigid_1 = np.stack(
            (
                [
                    f[index]["gripper_position_1"][k]
                    for k in range(0, self.frames_num - 1, dt)
                ]
            ),
            axis = 0
        )
        load_1 = np.stack(
            (
                [
                    f[index]["gripper_position_1"][k]
                    for k in range(dt, self.frames_num, dt)
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
                    f[index]["gripper_position_2"][k]
                    for k in range(0, self.frames_num - 1, dt)
                ]
            ),
            axis = 0
        )
        load_2 = np.stack(
            (
                [
                    f[index]["gripper_position_2"][k]
                    for k in range(dt, self.frames_num, dt)
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
                    f[index]["tissue_mesh_positions"][k]
                    for k in range(dt, self.frames_num, dt)
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
    

class DataLoader_Cavity_Grasping_LongTime(Dataset):
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
        self.id_paths = os.path.join(self.data_path, f"cavity_grasping_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        init_deformable_1 = np.array(f[index]["tissue_mesh_positions"][0])
        final_deformable_1 = np.array(f[index]["tissue_mesh_positions"][self.target_frame])
        target_deformable_1 = final_deformable_1 - init_deformable_1
        
        rigid_1 = np.array(f[index]["gripper_position_1"][0])
        load_1 = np.array(f[index]["gripper_position_1"][self.target_frame]) - rigid_1
        load_1 = np.concatenate(
            (rigid_1, load_1),
            axis = -1
        )

        rigid_2 = np.array(f[index]["gripper_position_2"][0])
        load_2 = np.array(f[index]["gripper_position_2"][self.target_frame]) - rigid_2
        load_2 = np.concatenate(
            (rigid_2, load_2),
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