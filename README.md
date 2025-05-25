# Unisoma (ICML 2025)

Code for "Unisoma: A Unified Transformer-based Solver for Multi-Solid Systems", accepted at International Conference on Machine Learning (ICML 2025).

![Overview](./assets/overview.png)

## Usage

### Install Python and Dependencies.

1. We recommend `Python 3.10`, `Pytorch 2.5.0`.

2. Install `torch_geometric` following the [official repo](https://github.com/pyg-team/pytorch_geometric).

3. Install other dependencies by:

```
pip install -r requirements.txt
```

### Prepare Data

You can download datasets from the [link]().

- Except for the Tissue Manipulation dataset, we preprocess other public datasets. You can download Tissue Manipulation dataset in [GGNS repo](https://github.com/jlinki/GGNS). 

- We provide self-curated datasets as well as certain preprocessed public datasets. If any copyright infringement is identified, please contact us at shilongtao@stu.pku.edu.cn, and we will address the issue promptly.

- The dataset should put in `input` folder with following file structure. Take `bilateral_stamping` as an example:

```
root_folder/
└──  bilateral_stamping/
    ├── assets/
    |   ├──checkpoints             # save model weight
    |   ├──logs                    # save training and test logs
    |   └──normalizers             # save normalizers
    └── data/          
        ├──input                   # save dataset
        └──ret                     # save prediction results
```

- Then modify the path of `assets` and `data` in the `xxx.toml`, where `xxx` indicates the dataset name.

### Train, Evaluation and Test

```
# Run in foreground
python exp_xxx.py

# Run in background
setsid python -u exp_xxx.py & 
```

- You can use the default config in the `configs` folder to reproduce the experiment results. And you can also explore more by modifying the configs.

- The output log is placed in the `logs` folder.

## Visualization

### Visualization of error maps.

<img src = "./assets/visualization.png" alt = "visualization" width = "800">

### Cavity Extruding rollout

<img src = "./assets/cavity_extruding.gif" alt = "cavity_extruding" width = "600">

### Cavity Grasping rollout

<img src = "./assets/cavity_grasping.gif" alt = "cavity_grasping" width = "600">

### Tissue Manipulation rollout

<img src = "./assets/tissue_manipulation.gif" alt = "tissue_manipulation" width = "600">


## Contact

Any further questions, please contact shilongtao@stu.pku.edu.cn.

## Citation

If you find this repo useful for you, please consider citing the following paper:
```
# TODO
```

## Acknowledgement

We sincerely appreciate the following repos for their code and dataset: 

- [Meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

- [GGNS](https://github.com/jlinki/GGNS)

- [DPI-Net](https://github.com/YunzhuLi/DPI-Net)

- [Neuraloperator](https://github.com/neuraloperator/neuraloperator)

- [Geo-FNO](https://github.com/neuraloperator/Geo-FNO)

- [Transolver](https://github.com/thuml/Transolver/tree/main)