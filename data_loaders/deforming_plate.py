import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Deforming_Plate_LongTime(Dataset):
    def __init__(
        self, 
        data_path: str,
        mode: str = "train",
        numeric_type = np.float32
    ):
        self.data_path = data_path
        self.mode = mode
        self.numeric_type = numeric_type
        self.id_paths = os.path.join(self.data_path, f"deforming_plate_dataset_{mode}.pkl")
        self.f = pickle.load(open(self.id_paths, "rb"))

    def __getitem__(self, index):
        f = self.f
        init_deformable_1 = np.array(f[index]["deformable_solid_pos"][0])

        rigid_1 = np.array(f[index]["rigid_solid_pos"][0])

        target_frame_id = -1
        max_load = -1
        for i in range(len(f[index]["rigid_solid_pos"])):
            current_load = np.sqrt(np.sum((rigid_1 - f[index]["rigid_solid_pos"][i]) ** 2, axis = -1)).mean()
            if current_load > max_load:
                max_load = current_load
                target_frame_id = i

        final_deformable_1 = np.array(f[index]["deformable_solid_pos"][target_frame_id])
        target_deformable_1 = final_deformable_1 - init_deformable_1
        load_1 = f[index]["rigid_solid_pos"][target_frame_id] - rigid_1
        load_1 = np.concatenate(
            (rigid_1, load_1),
            axis = -1
        )

        final_deformable_1_stress = np.array(f[index]["deformable_solid_stress"][target_frame_id]).reshape(-1, 1) / 1e3 # for stable training

        target_deformable_1 = np.concatenate(
            (
                target_deformable_1,
                final_deformable_1_stress
            ), axis = -1
        )

        final_deformable_1 = np.concatenate(
            (
                final_deformable_1,
                final_deformable_1_stress
            ), axis = -1
        )
        

        return [init_deformable_1.astype(self.numeric_type)], \
               [rigid_1.astype(self.numeric_type)], \
               [load_1.astype(self.numeric_type)], \
               [final_deformable_1.astype(self.numeric_type)], \
               [target_deformable_1.astype(self.numeric_type)]  

    def __len__(self):
        return len(self.f)