import os
import sys
import toml
import torch
import pickle

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import  CosineAnnealingLR
from data_loaders.cavity_extruding import DataLoader_Cavity_Extruding_Rollout
from model.unisoma import Unisoma
from model.loss import *
from utils.normalizers import Normalizer
from utils.edge_builder import knn_edge_builder

def train(data_path, net, config, normalizers, criterion):
    torch.manual_seed(config["rollout"]["hyperparameter"]["seed"])

    lr = config["rollout"]["hyperparameter"]["lr"]
    lr_decay = config["rollout"]["hyperparameter"]["lr_decay"]
    epochs = config["rollout"]["hyperparameter"]["epochs"]
    weight_decay = config["rollout"]["hyperparameter"]["weight_decay"]
    batch_size = config["rollout"]["hyperparameter"]["batch_size"]
    eval_per_epoch = config["rollout"]["hyperparameter"]["eval_per_epoch"]
    noise_sigma = config["rollout"]["hyperparameter"]["noise_sigma"]

    use_edge = config["rollout"]["unisoma"]["use_edge"]

    optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay)
    
    train_data_loader = DataLoader_Cavity_Extruding_Rollout(
        data_path = data_path,
        mode = "train"
    )

    eval_data_loader = DataLoader_Cavity_Extruding_Rollout(
        data_path = data_path,
        mode = "eval"
    )

    train_datas = DataLoader(
        train_data_loader,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    )

    eval_datas = DataLoader(
        eval_data_loader,
        batch_size = 1,
        num_workers = 0
    )

    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_datas), eta_min = lr / lr_decay)

    net.to(device)

    # normalization
    for i, train_data in enumerate(train_datas):
        current_deformable, rigid, load, _, target_deformable = train_data
        total_batch = current_deformable[0].shape[1]
        for b in range(0, total_batch, batch_size):
            _current_deformable = [
                d[0, b : b + batch_size].to(device)
                for d in current_deformable
            ]
            if use_edge:
                edges = [
                    knn_edge_builder(d, k = config["rollout"]["unisoma"]["k"])
                    for d in _current_deformable
                ]
                for j in range(len(edges)):
                    normalizers["edges"][f"current_deformable_{j}"](edges[j][0], accumulate = True)
            for j in range(len(current_deformable)):
                normalizers["points"][f"current_deformable_{j}"](_current_deformable[j], accumulate = True)
            for j in range(len(rigid)):
                normalizers["points"][f"rigid_{j}"](rigid[j][0, b : b + batch_size].to(device), accumulate = True)
            for j in range(len(load)):
                normalizers["points"][f"load_{j}"](load[j][0, b : b + batch_size].to(device), accumulate = True)
            for j in range(len(target_deformable)):
                normalizers["points"][f"target_deformable_{j}"](target_deformable[j][0, b : b + batch_size].to(device), accumulate = True)

    for key_1, value_1 in normalizers.items():
        for key_2, value_2 in value_1.items():
            value_2.save_variable(f"{assets_root}/normalizers/{model_name}_rollout_{key_1}_{key_2}_normalizer_{version}.pth")

    min_eval_loss = 1e10
    for epoch in range(epochs):
        net.train()
        mean_train_loss = 0.
        mean_train_losses = [0.] * len(current_deformable)
        batch_num = 0
        for i, train_data in enumerate(train_datas):
            current_deformable, rigid, load, _, target_deformable = train_data
            total_batch = current_deformable[0].shape[1]
            for b in range(0, total_batch, batch_size):
                _current_deformable = [
                    d.squeeze(0)[b : b + batch_size].to(device)
                    for d in current_deformable
                ]
                if use_edge:
                    edges = [
                        knn_edge_builder(d, k = config["rollout"]["unisoma"]["k"])
                        for d in _current_deformable
                    ]
                    for j in range(len(edges)):
                        edges[j][0] = normalizers["edges"][f"current_deformable_{j}"](edges[j][0])
                for j in range(len(current_deformable)):
                    _current_deformable[j] = normalizers["points"][f"current_deformable_{j}"](_current_deformable[j])
                _rigid, _load, _target_deformable = [], [], []
                for j in range(len(rigid)):
                    _rigid.append(normalizers["points"][f"rigid_{j}"](rigid[j][0, b : b + batch_size].to(device)))
                for j in range(len(load)):
                    _load.append(normalizers["points"][f"load_{j}"](load[j][0, b : b + batch_size].to(device)))
                for j in range(len(target_deformable)):
                    _target_deformable.append(normalizers["points"][f"target_deformable_{j}"](target_deformable[j][0, b : b + batch_size].to(device)))
                _noised_current_deformable = [
                    torch.randn_like(_current_deformable[j]) * noise_sigma + _current_deformable[j]
                    for j in range(len(_current_deformable))
                ]

                outputs = net(
                    _load, _rigid, _noised_current_deformable, edges if use_edge else None
                )

                losses = [criterion(outputs[j], _target_deformable[j]) for j in range(len(target_deformable))]
                loss = sum(losses)

                print(f"epoch: {epoch}, lr: {scheduler.get_last_lr()[0]}, sample {i}, batch: {b}, train_loss: {loss.data}, ", end = "")
                for j in range(len(losses)):
                    print(f"train_loss_{j}: {losses[j].data}, ", end = "")
                    if j == len(losses) - 1: print()
                    mean_train_losses[j] += losses[j].data
                mean_train_loss += loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_num += 1
            scheduler.step()
        mean_train_loss /= batch_num
        mean_train_losses = [mtl / len(train_datas) for mtl in mean_train_losses]
        print(f"epoch: {epoch}, mean_train_loss: {mean_train_loss}, ", end = "")
        for j in range(len(mean_train_losses)):
            print(f"mean_train_loss_{j}: {mean_train_losses[j].data}, ", end = "")
            if j == len(mean_train_losses) - 1: print()

        if (epoch + 1) % eval_per_epoch != 0: continue
        net.eval()
        mean_eval_loss = 0.
        mean_eval_losses = [0.] * len(current_deformable)
        with torch.no_grad():
            for i, eval_data in enumerate(eval_datas):
                current_deformable, rigid, load, next_deformable, _ = eval_data
                total_batch = current_deformable[0].shape[1]
                mean_eval_trajectory_loss = 0.
                mean_eval_trajectory_losses = [0.] * len(current_deformable)
                pred_pos = None
                for b in range(total_batch):
                    if b == 0:
                        _current_deformable = [
                            d[0, b : b + 1].to(device)
                            for d in current_deformable
                        ]
                    else:
                        _current_deformable = pred_pos

                    if use_edge:
                        edges = [
                            knn_edge_builder(d, k = config["rollout"]["unisoma"]["k"])
                            for d in _current_deformable
                        ]
                        for j in range(len(edges)):
                            edges[j][0] = normalizers["edges"][f"current_deformable_{j}"](edges[j][0])
                    for j in range(len(current_deformable)):
                        _current_deformable[j] = normalizers["points"][f"current_deformable_{j}"](_current_deformable[j])
                    _rigid, _load, _next_deformable = [], [], []
                    for j in range(len(rigid)):
                        _rigid.append(normalizers["points"][f"rigid_{j}"](rigid[j][0, b : b + 1].to(device)))
                    for j in range(len(load)):
                        _load.append(normalizers["points"][f"load_{j}"](load[j][0, b : b + 1].to(device)))
                    for j in range(len(next_deformable)):
                        _next_deformable.append(next_deformable[j][0, b : b + 1].to(device))
                    outputs = net(
                        _load, _rigid, _current_deformable, edges if use_edge else None
                    )
                    pred_pos = [
                        normalizers["points"][f"current_deformable_{j}"].inverse(_current_deformable[j]) + \
                        normalizers["points"][f"target_deformable_{j}"].inverse(outputs[j])
                        for j in range(len(outputs))
                    ]

                    losses = [criterion(pred_pos[j], _next_deformable[j], use_sqrt = True) for j in range(len(pred_pos))]
                    loss = sum(losses)
                    print(f"epoch: {epoch}, sample: {i}, frame: {b}, eval_loss: {loss.data}, ", end = "")
                    mean_eval_trajectory_loss += loss.data
                    for j in range(len(losses)):
                        print(f"eval_loss_{j}: {losses[j].data}, ", end = "")
                        if j == len(losses) - 1: print()
                        mean_eval_trajectory_losses[j] += losses[j].data
                
                mean_eval_trajectory_loss /= total_batch
                mean_eval_trajectory_losses = [metl / total_batch for metl in mean_eval_trajectory_losses]
                mean_eval_loss += mean_eval_trajectory_loss
                mean_eval_losses = [mel + metl for mel, metl in zip(mean_eval_losses, mean_eval_trajectory_losses)]
                print(f"epoch: {epoch}, sample: {i}, mean_eval_trajectory_loss: {mean_eval_trajectory_loss}, ", end = "")
                for j in range(len(mean_eval_trajectory_losses)):
                    print(f"mean_eval_trajectory_loss_{j}: {mean_eval_trajectory_losses[j].data}, ", end = "")
                    if j == len(mean_eval_trajectory_losses) - 1: print()

            mean_eval_loss /= len(eval_datas)
            mean_eval_losses = [mel / len(eval_datas) for mel in mean_eval_losses]
            print(f"epoch: {epoch}, mean_eval_loss: {mean_eval_loss}")
            for j in range(len(mean_eval_losses)):
                print(f"mean_eval_loss_{j}: {mean_eval_losses[j].data}, ", end = "")
                if j == len(mean_eval_losses) - 1: print()
            if mean_eval_loss < min_eval_loss:
                min_eval_loss = mean_eval_loss
                torch.save(
                    net.state_dict(), 
                    f"{assets_root}/checkpoints/{model_name}_rollout_best_model_{version}.pth"
                )

def test(data_path, net, config, normalizers, criterion):
    ret_save_path = f"{data_root}/ret/{model_name}_cavity_extruding_rollout_{version}.pkl"
    test_data_loader = DataLoader_Cavity_Extruding_Rollout(
        data_path = data_path,
        mode = "test"
    )

    use_edge = config["rollout"]["unisoma"]["use_edge"]

    test_datas = DataLoader(
        test_data_loader,
        batch_size = 1,
        num_workers = 0
    )
    net.to(device)
    net.load_state_dict(torch.load(f"{assets_root}/checkpoints/{model_name}_rollout_best_model_{version}.pth", weights_only = True, map_location = device))
    for key_1, value_1 in normalizers.items():
        for key_2, value_2 in value_1.items():
            value_2.load_variable(f"{assets_root}/normalizers/{model_name}_rollout_{key_1}_{key_2}_normalizer_{version}.pth")
    net.eval()

    mean_test_loss = 0.
    mean_test_losses = [0.] * config["rollout"]["unisoma"]["deformable_num"]
    with torch.no_grad():
        ret_save_data = {}
        for i, test_data in enumerate(test_datas):
            current_deformable, rigid, load, next_deformable, _ = test_data
            total_batch = current_deformable[0].shape[1]
            mean_test_trajectory_loss = 0.
            mean_test_trajectory_losses = [0.] * len(current_deformable)
            pred_pos = None
            trajectory = []
            ret_save_data[i] = {}
            for b in range(total_batch):
                if b == 0:
                    _current_deformable = [
                        d[0, b : b + 1].to(device)
                        for d in current_deformable
                    ]
                else:
                    _current_deformable = pred_pos

                if use_edge:
                    edges = [
                        knn_edge_builder(d, k = config["rollout"]["unisoma"]["k"])
                        for d in _current_deformable
                    ]
                    for j in range(len(edges)):
                        edges[j][0] = normalizers["edges"][f"current_deformable_{j}"](edges[j][0])
                for j in range(len(current_deformable)):
                    _current_deformable[j] = normalizers["points"][f"current_deformable_{j}"](_current_deformable[j])
                _rigid, _load, _next_deformable = [], [], []
                for j in range(len(rigid)):
                    _rigid.append(normalizers["points"][f"rigid_{j}"](rigid[j][0, b : b + 1].to(device)))
                for j in range(len(load)):
                    _load.append(normalizers["points"][f"load_{j}"](load[j][0, b : b + 1].to(device)))
                for j in range(len(next_deformable)):
                    _next_deformable.append(next_deformable[j][0, b : b + 1].to(device))
                outputs = net(
                    _load, _rigid, _current_deformable, edges if use_edge else None
                )
                pred_pos = [
                    normalizers["points"][f"current_deformable_{j}"].inverse(_current_deformable[j]) + \
                    normalizers["points"][f"target_deformable_{j}"].inverse(outputs[j])
                    for j in range(len(outputs))
                ]

                losses = [criterion(pred_pos[j], _next_deformable[j], use_sqrt = True) for j in range(len(pred_pos))]
                loss = sum(losses)
                print(f"sample: {i}, frame: {b}, test_loss: {loss.data}, ", end = "")
                mean_test_trajectory_loss += loss.data
                for j in range(len(losses)):
                    print(f"test_loss_{j}: {losses[j].data}, ", end = "")
                    if j == len(losses) - 1: print()
                    mean_test_trajectory_losses[j] += losses[j].data
                trajectory.append([p.cpu().numpy().copy() for p in pred_pos])
            mean_test_trajectory_loss /= total_batch
            mean_test_trajectory_losses = [mttl / total_batch for mttl in mean_test_trajectory_losses]
            mean_test_loss += mean_test_trajectory_loss
            mean_test_losses = [mtl + mttl for mtl, mttl in zip(mean_test_losses, mean_test_trajectory_losses)]
            print(f"sample: {i}, mean_test_trajectory_loss: {mean_test_trajectory_loss}, ", end = "")
            for j in range(len(mean_test_trajectory_losses)):
                print(f"mean_test_trajectory_loss_{j}: {mean_test_trajectory_losses[j].data}, ", end = "")
                if j == len(mean_test_trajectory_losses) - 1: print()
        mean_test_loss /= len(test_datas)
        mean_test_losses = [mtl / len(test_datas) for mtl in mean_test_losses]
        print(f"mean_test_loss: {mean_test_loss}, ", end = "")
        for j in range(len(mean_test_losses)):
            print(f"mean_test_loss_{j}: {mean_test_losses[j].data}, ", end = "")
            if j == len(mean_test_losses) - 1: print()
        
    pickle.dump(ret_save_data, open(ret_save_path, "wb"))


def main(run_type = "train"):
    data_path = f"{data_root}/input"
    unisoma_params = config["rollout"]["unisoma"]
    net = Unisoma(
        emb_dim = unisoma_params["emb_dim"],
        slice_num = unisoma_params["slice_num"],
        use_edge = unisoma_params["use_edge"],
        load_dim = unisoma_params["load_dim"],
        rigid_dim = unisoma_params["rigid_dim"],
        input_deformable_dim = unisoma_params["input_deformable_dim"],
        output_deformable_dim = unisoma_params["output_deformable_dim"],
        load_num = unisoma_params["load_num"],
        rigid_num = unisoma_params["rigid_num"],
        deformable_num = unisoma_params["deformable_num"],
        contact_id = unisoma_params["contact_id"],
        processor_num = unisoma_params["processor_num"],
        attention_heads_num = unisoma_params["attention_heads_num"],
        mlp_ratio = unisoma_params["mlp_ratio"],
        dropout = unisoma_params["dropout"],
        bias = unisoma_params["bias"],
        act = unisoma_params["act"]
    )

    if config["rollout"]["hyperparameter"]["criterion"] == "relative l2":
        criterion = relative_l2_loss
    elif config["rollout"]["hyperparameter"]["criterion"] == "mse l2":
        criterion = mse_l2_loss
    else:
        raise NotImplementedError

    normalizers = {
        "points": {},
        "edges": {}
    }

    for obj in ["load", "rigid", "current_deformable", "target_deformable"]:
        if obj in ["load", "rigid"]: 
            size = unisoma_params[f"{obj}_dim"]
            num = unisoma_params[f"{obj}_num"]
        elif obj == "current_deformable": 
            size = unisoma_params[f"input_deformable_dim"]
            num = unisoma_params["deformable_num"]
        else: 
            size = unisoma_params[f"output_deformable_dim"]
            num = unisoma_params["deformable_num"]
        for i in range(num):
            normalizers["points"][f"{obj}_{i}"] = Normalizer(
                size = size,
                name = f"{obj}_{i}_points_normalizer",
                device = device
            )
    
    for i in range(unisoma_params["deformable_num"]):
        normalizers["edges"][f"current_deformable_{i}"] = Normalizer(
            size = 1,
            name = f"current_deformable_{i}_edges_normalizer",
            device = device
        )

    if run_type == "train":
        train(data_path, net, config, normalizers, criterion)
    elif run_type == "test":
        test(data_path, net, config, normalizers, criterion)


if __name__ == "__main__":
    config_path = "./configs/cavity_extruding.toml"
    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = toml.load(f)

    version = config["rollout"]["version"]
    assets_root = config["rollout"]["assets_root"]
    data_root = config["rollout"]["data_root"]
    model_name = config["rollout"]["model_name"]

    device = config["rollout"]["hyperparameter"]["device"]
    log_path = f"{assets_root}/logs/{model_name}_rollout_log_{version}.log"
    log_writer = open(log_path, 'a', encoding = "utf8")

    sys.stdout = log_writer
    sys.stderr = log_writer

    print(config["rollout"]["description"])

    main("train")
    main("test")

    log_writer.close()

    # setsid python -u exp_cavity_extruding_rollout.py &