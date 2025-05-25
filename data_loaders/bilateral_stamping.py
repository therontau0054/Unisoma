import os
import copy
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Bilateral_Stamping_LongTime(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"bilateral_stamping_dataset_{self.mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        init_deformable_1 = f[index]["init_metal_pos"]
        init_deformable_2 = f[index]["init_rubber_pos"]
        rigid_1 = f[index]["die_1_pos"]
        rigid_2 = f[index]["die_2_pos"]
        final_deformable_1 = f[index]["final_metal_pos"]
        final_deformable_2 = f[index]["final_rubber_pos"]
        load_1 = copy.deepcopy(rigid_1)
        load_1[:, 3:] += load_1[:, :3]
        load_2 = copy.deepcopy(rigid_2)
        load_2[:, 3:] += load_2[:, :3]
        target_deformable_1 = f[index]["final_metal_pos"] - f[index]["init_metal_pos"]
        target_deformable_2 = f[index]["final_rubber_pos"] - f[index]["init_rubber_pos"]
        target_deformable_1_stress = f[index]["final_metal_stress"].reshape(-1, 1)
        target_deformable_1_peeq = f[index]["final_metal_peeq"].reshape(-1, 1)
        target_deformable_2_stress = f[index]["final_rubber_stress"].reshape(-1, 1)
        target_deformable_2_peeq = f[index]["final_rubber_peeq"].reshape(-1, 1)

        
        final_deformable_1 = np.concatenate(
            (
                final_deformable_1,
                target_deformable_1_stress,
                target_deformable_1_peeq
            ), axis = -1
        )

        target_deformable_1 = np.concatenate(
            (
                target_deformable_1,
                target_deformable_1_stress,
                target_deformable_1_peeq
            ), axis = -1
        )

        final_deformable_2 = np.concatenate(
            (
                final_deformable_2,
                target_deformable_2_stress,
                target_deformable_2_peeq
            ), axis = -1
        )

        target_deformable_2 = np.concatenate(
            (
                target_deformable_2,
                target_deformable_2_stress,
                target_deformable_2_peeq
            ), axis = -1
        )


        return [
                   init_deformable_1.astype(self.numeric_type),
                   init_deformable_2.astype(self.numeric_type)
               ], \
               [
                   rigid_1.astype(self.numeric_type)[:, :3],
                   rigid_2.astype(self.numeric_type)[:, :3]
               ], \
               [
                   load_1.astype(self.numeric_type),
                   load_2.astype(self.numeric_type)
               ], \
               [
                   final_deformable_1.astype(self.numeric_type),
                   final_deformable_2.astype(self.numeric_type)
               ], \
               [
                   target_deformable_1.astype(self.numeric_type),
                   target_deformable_2.astype(self.numeric_type)
               ]
    
    def __len__(self):
        return len(self.f)