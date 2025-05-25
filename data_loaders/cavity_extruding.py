import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Cavity_Extruding_Rollout(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        frames_num: int = 121,
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.frames_num = frames_num
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"cavity_extruding_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        current_deformable_1 = np.stack(
            (
                [
                    f[index]["cavity_1_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        current_deformable_2 = np.stack(
            (
                [
                    f[index]["cavity_2_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        current_deformable_3 = np.stack(
            (
                [
                    f[index]["cavity_3_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        ##############################################################
        rigid_1 = np.stack(
            (
                [
                    f[index]["gripper_1_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        load_1 = np.stack(
            (
                [
                    f[index]["gripper_1_pos"][k]
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


        ##############################################################
        rigid_2 = np.stack(
            (
                [
                    f[index]["gripper_2_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        load_2 = np.stack(
            (
                [
                    f[index]["gripper_2_pos"][k]
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


        ##############################################################
        rigid_3 = np.stack(
            (
                [
                    f[index]["gripper_3_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        load_3 = np.stack(
            (
                [
                    f[index]["gripper_3_pos"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        load_3 = load_3 - rigid_3
        load_3 = np.concatenate(
            (rigid_3, load_3),
            axis = -1
        )


        ##############################################################
        rigid_4 = np.stack(
            (
                [
                    f[index]["gripper_4_pos"][k]
                    for k in range(self.frames_num - 1)
                ]
            ),
            axis = 0
        )

        load_4 = np.stack(
            (
                [
                    f[index]["gripper_4_pos"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        load_4 = load_4 - rigid_4
        load_4 = np.concatenate(
            (rigid_4, load_4),
            axis = -1
        )

        ##############################################################

        next_deformable_1 = np.stack(
            (
                [
                    f[index]["cavity_1_pos"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        target_deformable_1 = next_deformable_1 - current_deformable_1

        ##############################################################

        next_deformable_2 = np.stack(
            (
                [
                    f[index]["cavity_2_pos"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        target_deformable_2 = next_deformable_2 - current_deformable_2


        ##############################################################

        next_deformable_3 = np.stack(
            (
                [
                    f[index]["cavity_3_pos"][k]
                    for k in range(1, self.frames_num)
                ]
            ),
            axis = 0
        )
        target_deformable_3 = next_deformable_3 - current_deformable_3

            
        return [
                   current_deformable_1.astype(self.numeric_type),
                   current_deformable_2.astype(self.numeric_type),
                   current_deformable_3.astype(self.numeric_type)
               ], \
               [
                   rigid_1.astype(self.numeric_type),
                   rigid_2.astype(self.numeric_type),
                   rigid_3.astype(self.numeric_type),
                   rigid_4.astype(self.numeric_type)
               ], \
               [
                   load_1.astype(self.numeric_type),
                   load_2.astype(self.numeric_type),
                   load_3.astype(self.numeric_type),
                   load_4.astype(self.numeric_type)
               ], \
               [
                   next_deformable_1.astype(self.numeric_type),
                   next_deformable_2.astype(self.numeric_type),
                   next_deformable_3.astype(self.numeric_type)
               ], \
               [
                   target_deformable_1.astype(self.numeric_type),
                   target_deformable_2.astype(self.numeric_type),
                   target_deformable_3.astype(self.numeric_type)
               ] 

    def __len__(self):
        return len(self.f)