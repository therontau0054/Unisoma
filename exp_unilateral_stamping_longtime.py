import os
import sys
import toml
import torch
import pickle

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import  CosineAnnealingLR
from data_loaders.unilateral_stamping import DataLoader_Unilateral_Stamping_LongTime
from model.unisoma import Unisoma
from model.loss import *
from utils.normalizers import Normalizer
from utils.edge_builder import knn_edge_builder

def train(data_path, net, config, normalizers, criterion):
    torch.manual_seed(config["longtime"]["hyperparameter"]["seed"])

    lr = config["longtime"]["hyperparameter"]["lr"]
    lr_decay = config["longtime"]["hyperparameter"]["lr_decay"]
    epochs = config["longtime"]["hyperparameter"]["epochs"]
    weight_decay = config["longtime"]["hyperparameter"]["weight_decay"]
    batch_size = config["longtime"]["hyperparameter"]["batch_size"]
    eval_per_epoch = config["longtime"]["hyperparameter"]["eval_per_epoch"]

    use_edge = config["longtime"]["unisoma"]["use_edge"]

    optimizer = torch.optim.AdamW(net.parameters(), lr = lr, weight_decay = weight_decay)
    
    train_data_loader = DataLoader_Unilateral_Stamping_LongTime(
        data_path = data_path,
        mode = "train"
    )

    eval_data_loader = DataLoader_Unilateral_Stamping_LongTime(
        data_path = data_path,
        mode = "eval"
    )

    train_datas = DataLoader(
        train_data_loader,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    eval_datas = DataLoader(
        eval_data_loader,
        batch_size = batch_size,
        num_workers = 0
    )

    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_datas), eta_min = lr / lr_decay)

    net.to(device)

    # normalization
    for i, train_data in enumerate(train_datas):
        init_deformable, rigid, load, _, target_deformable = train_data
        
        if use_edge:
            edges = [
                knn_edge_builder(d.to(device), k = config["longtime"]["unisoma"]["k"])
                for d in init_deformable
            ]
            for j in range(len(edges)):
                normalizers["edges"][f"init_deformable_{j}"](edges[j][0].to(device), accumulate = True)

        for j in range(len(init_deformable)):
            normalizers["points"][f"init_deformable_{j}"](init_deformable[j].to(device), accumulate = True)
        for j in range(len(rigid)):
            normalizers["points"][f"rigid_{j}"](rigid[j].to(device), accumulate = True)
        for j in range(len(load)):
            normalizers["points"][f"load_{j}"](load[j].to(device), accumulate = True)
        for j in range(len(target_deformable)):
            normalizers["points"][f"target_deformable_{j}"](target_deformable[j].to(device), accumulate = True)

    for key_1, value_1 in normalizers.items():
        for key_2, value_2 in value_1.items():
            value_2.save_variable(f"{assets_root}/normalizers/{model_name}_longtime_{key_1}_{key_2}_normalizer_{version}.pth")

    min_eval_loss = 1e10
    for epoch in range(epochs):
        net.train()
        mean_train_loss = 0.
        mean_train_pos_losses = [0.] * len(init_deformable)
        mean_train_stress_losses = [0.] * len(init_deformable)
        mean_train_peeq_losses = [0.] * len(init_deformable)
        for i, train_data in enumerate(train_datas):
            init_deformable, rigid, load, _, target_deformable = train_data
            
            if use_edge:
                edges = [
                    knn_edge_builder(d.to(device), k = config["longtime"]["unisoma"]["k"])
                    for d in init_deformable
                ]
                for j in range(len(edges)):
                    edges[j][0] = normalizers["edges"][f"init_deformable_{j}"](edges[j][0].to(device))

            for j in range(len(init_deformable)):
                init_deformable[j] = normalizers["points"][f"init_deformable_{j}"](init_deformable[j].to(device))
            for j in range(len(rigid)):
                rigid[j] = normalizers["points"][f"rigid_{j}"](rigid[j].to(device))
            for j in range(len(load)):
                load[j] = normalizers["points"][f"load_{j}"](load[j].to(device))
            for j in range(len(target_deformable)):
                target_deformable[j] = normalizers["points"][f"target_deformable_{j}"](target_deformable[j].to(device))
            outputs = net(
                load, rigid, init_deformable, edges if use_edge else None
            )
            pos_losses = [criterion(outputs[j][:, :, :3], target_deformable[j][:, :, :3]) for j in range(len(target_deformable))]
            stress_losses = [criterion(outputs[j][:, :, 3:4], target_deformable[j][:, :, 3:4]) for j in range(len(target_deformable))]
            peeq_losses = [criterion(outputs[j][:, :, 4:], target_deformable[j][:, :, 4:]) for j in range(len(target_deformable))]
            pos_loss = sum(pos_losses)
            stress_loss = sum(stress_losses)
            peeq_loss = sum(peeq_losses)
            loss = pos_loss + stress_loss + peeq_loss
            print(f"epoch: {epoch}, lr: {scheduler.get_last_lr()[0]}, batch: {i}, train_loss: {loss.data}, ", end = "")
            for j in range(len(pos_losses)):
                print(f"train_pos_loss_{j}: {pos_losses[j].data}, ", end = "")
                print(f"train_stress_loss_{j}: {stress_losses[j].data}, ", end = "")
                print(f"train_peeq_loss_{j}: {peeq_losses[j].data}", end = "")
                if j == len(pos_losses) - 1: print()
                mean_train_pos_losses[j] += pos_losses[j].data
                mean_train_stress_losses[j] += stress_losses[j].data
                mean_train_peeq_losses[j] += peeq_losses[j].data
            mean_train_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        mean_train_loss /= len(train_datas)
        mean_train_pos_losses = [mtpl / len(train_datas) for mtpl in mean_train_pos_losses]
        mean_train_stress_losses = [mtsl / len(train_datas) for mtsl in mean_train_stress_losses]
        mean_train_peeq_losses = [mtpl / len(train_datas) for mtpl in mean_train_peeq_losses]
        print(f"epoch: {epoch}, mean_train_loss: {mean_train_loss}, ", end = "")
        for j in range(len(mean_train_pos_losses)):
                print(f"mean_train_pos_loss_{j}: {mean_train_pos_losses[j].data}, ", end = "")
                print(f"mean_train_stress_loss_{j}: {mean_train_stress_losses[j].data}, ", end = "")
                print(f"mean_train_peeq_loss_{j}: {mean_train_peeq_losses[j].data}", end = "")
                if j == len(mean_train_pos_losses) - 1: print()
        if (epoch + 1) % eval_per_epoch != 0: continue
        net.eval()
        mean_eval_loss = 0.
        mean_eval_pos_losses = [0.] * len(init_deformable)
        mean_eval_stress_losses = [0.] * len(init_deformable)
        mean_eval_peeq_losses = [0.] * len(init_deformable)
        with torch.no_grad():
            for i, eval_data in enumerate(eval_datas):
                init_deformable, rigid, load, final_deformable, _ = eval_data
            
                if use_edge:
                    edges = [
                        knn_edge_builder(d.to(device), k = config["longtime"]["unisoma"]["k"])
                        for d in init_deformable
                    ]
                    for j in range(len(edges)):
                        edges[j][0] = normalizers["edges"][f"init_deformable_{j}"](edges[j][0].to(device))

                for j in range(len(init_deformable)):
                    init_deformable[j] = normalizers["points"][f"init_deformable_{j}"](init_deformable[j].to(device))
                for j in range(len(rigid)):
                    rigid[j] = normalizers["points"][f"rigid_{j}"](rigid[j].to(device))
                for j in range(len(load)):
                    load[j] = normalizers["points"][f"load_{j}"](load[j].to(device))
                for j in range(len(final_deformable)):
                    final_deformable[j] = final_deformable[j].to(device)
                outputs = net(
                    load, rigid, init_deformable, edges if use_edge else None
                )
                outputs = [
                    normalizers["points"][f"target_deformable_{j}"].inverse(outputs[j])
                    for j in range(len(outputs))
                ]
                pred_pos = [
                    normalizers["points"][f"init_deformable_{j}"].inverse(init_deformable[j]) + outputs[j][:, :, :3]
                    for j in range(len(outputs))
                ]
                pos_losses = [criterion(pred_pos[j], final_deformable[j][:, :, :3]) for j in range(len(final_deformable))]
                stress_losses = [criterion(outputs[j][:, :, 3:4], final_deformable[j][:, :, 3:4]) for j in range(len(final_deformable))]
                peeq_losses = [criterion(outputs[j][:, :, 4:], final_deformable[j][:, :, 4:]) for j in range(len(final_deformable))]
                pos_loss = sum(pos_losses)
                stress_loss = sum(stress_losses)
                peeq_loss = sum(peeq_losses)
                loss = pos_loss + stress_loss + peeq_loss
                print(f"epoch: {epoch}, batch: {i}, eval_loss: {loss.data}, ", end = "")
                for j in range(len(pos_losses)):
                    print(f"eval_pos_loss_{j}: {pos_losses[j].data}, ", end = "")
                    print(f"eval_stress_loss_{j}: {stress_losses[j].data}, ", end = "")
                    print(f"eval_peeq_loss_{j}: {peeq_losses[j].data}", end = "")
                    if j == len(pos_losses) - 1: print()
                    mean_eval_pos_losses[j] += pos_losses[j].data
                    mean_eval_stress_losses[j] += stress_losses[j].data
                    mean_eval_peeq_losses[j] += peeq_losses[j].data
                mean_eval_loss += loss.data

            mean_eval_loss /= len(eval_datas)
            mean_eval_pos_losses = [mepl / len(eval_datas) for mepl in mean_eval_pos_losses]
            mean_eval_stress_losses = [mesl / len(eval_datas) for mesl in mean_eval_stress_losses]
            mean_eval_peeq_losses = [mepl / len(eval_datas) for mepl in mean_eval_peeq_losses]
            print(f"epoch: {epoch}, mean_eval_loss: {mean_eval_loss}, ", end = "")
            for j in range(len(mean_eval_pos_losses)):
                print(f"mean_eval_pos_loss_{j}: {mean_eval_pos_losses[j].data}, ", end = "")
                print(f"mean_eval_stress_loss_{j}: {mean_eval_stress_losses[j].data}, ", end = "")
                print(f"mean_eval_peeq_loss_{j}: {mean_eval_peeq_losses[j].data}", end = "")
                if j == len(mean_eval_pos_losses) - 1: print()
            if mean_eval_loss < min_eval_loss:
                min_eval_loss = mean_eval_loss
                torch.save(
                    net.state_dict(), 
                    f"{assets_root}/checkpoints/{model_name}_longtime_best_model_{version}.pth"
                )


def test(data_path, net, config, normalizers, criterion):
    ret_save_path = f"{data_root}/ret/{model_name}_unilateral_stamping_longtime_{version}.pkl"
    test_data_loader = DataLoader_Unilateral_Stamping_LongTime(
        data_path = data_path,
        mode = "test"
    )

    batch_size = config["longtime"]["hyperparameter"]["batch_size"]

    use_edge = config["longtime"]["unisoma"]["use_edge"]

    test_datas = DataLoader(
        test_data_loader,
        batch_size = batch_size,
        num_workers = 0
    )
    net.to(device)
    net.load_state_dict(torch.load(f"{assets_root}/checkpoints/{model_name}_longtime_best_model_{version}.pth", weights_only = True, map_location = device))
    for key_1, value_1 in normalizers.items():
        for key_2, value_2 in value_1.items():
            value_2.load_variable(f"{assets_root}/normalizers/{model_name}_longtime_{key_1}_{key_2}_normalizer_{version}.pth")
    net.eval()

    mean_test_loss = 0.
    mean_test_pos_losses = [0.] * config["longtime"]["unisoma"]["deformable_num"]
    mean_test_stress_losses = [0.] * config["longtime"]["unisoma"]["deformable_num"]
    mean_test_peeq_losses = [0.] * config["longtime"]["unisoma"]["deformable_num"]
    with torch.no_grad():
        ret_save_data = {}
        for i, test_data in enumerate(test_datas):
            init_deformable, rigid, load, final_deformable, _ = test_data
            ret_save_data[i] = {}
            if use_edge:
                edges = [
                    knn_edge_builder(d.to(device), k = config["longtime"]["unisoma"]["k"])
                    for d in init_deformable
                ]
                for j in range(len(edges)):
                    edges[j][0] = normalizers["edges"][f"init_deformable_{j}"](edges[j][0].to(device))

            for j in range(len(init_deformable)):
                init_deformable[j] = normalizers["points"][f"init_deformable_{j}"](init_deformable[j].to(device))
            for j in range(len(rigid)):
                rigid[j] = normalizers["points"][f"rigid_{j}"](rigid[j].to(device))
            for j in range(len(load)):
                load[j] = normalizers["points"][f"load_{j}"](load[j].to(device))
            for j in range(len(final_deformable)):
                final_deformable[j] = final_deformable[j].to(device)
            outputs = net(
                load, rigid, init_deformable, edges if use_edge else None
            )
            outputs = [
                normalizers["points"][f"target_deformable_{j}"].inverse(outputs[j])
                for j in range(len(outputs))
            ]
            pred_pos = [
                normalizers["points"][f"init_deformable_{j}"].inverse(init_deformable[j]) + outputs[j][:, :, :3]
                for j in range(len(outputs))
            ]
            pos_losses = [criterion(pred_pos[j], final_deformable[j][:, :, :3]) for j in range(len(final_deformable))]
            stress_losses = [criterion(outputs[j][:, :, 3:4], final_deformable[j][:, :, 3:4]) for j in range(len(final_deformable))]
            peeq_losses = [criterion(outputs[j][:, :, 4:], final_deformable[j][:, :, 4:]) for j in range(len(final_deformable))]
            pos_loss = sum(pos_losses)
            stress_loss = sum(stress_losses)
            peeq_loss = sum(peeq_losses)
            loss = pos_loss + stress_loss + peeq_loss
            print(f"batch: {i}, test_loss: {loss.data}, ", end = "")
            for j in range(len(pos_losses)):
                print(f"test_pos_loss_{j}: {pos_losses[j].data}, ", end = "")
                print(f"test_stress_loss_{j}: {stress_losses[j].data}, ", end = "")
                print(f"test_peeq_loss_{j}: {peeq_losses[j].data}", end = "")
                if j == len(pos_losses) - 1: print()
                mean_test_pos_losses[j] += pos_losses[j].data
                mean_test_stress_losses[j] += stress_losses[j].data
                mean_test_peeq_losses[j] += peeq_losses[j].data
            mean_test_loss += loss.data
            ret_save_data[i]["pred_pos"] = [p.cpu().numpy().copy() for p in pred_pos]
            ret_save_data[i]["pred_stress"] = [o[:, :, 3:4].cpu().numpy().copy() for o in outputs]
            ret_save_data[i]["pred_peeq"] = [o[:, :, 4:].cpu().numpy().copy() for o in outputs]
        
        mean_test_loss /= len(test_datas)
        mean_test_pos_losses = [mtpl / len(test_datas) for mtpl in mean_test_pos_losses]
        mean_test_stress_losses = [mtsl / len(test_datas) for mtsl in mean_test_stress_losses]
        mean_test_peeq_losses = [mtpl / len(test_datas) for mtpl in mean_test_peeq_losses]
        print(f"mean_test_loss: {mean_test_loss}, ", end = "")
        for j in range(len(mean_test_pos_losses)):
            print(f"mean_test_pos_loss_{j}: {mean_test_pos_losses[j].data}, ", end = "")
            print(f"mean_test_stress_loss_{j}: {mean_test_stress_losses[j].data}, ", end = "")
            print(f"mean_test_peeq_loss_{j}: {mean_test_peeq_losses[j].data}", end = "")
            if j == len(mean_test_pos_losses) - 1: print()
    pickle.dump(ret_save_data, open(ret_save_path, "wb"))


def main(run_type = "train"):
    data_path = f"{data_root}/input"
    unisoma_params = config["longtime"]["unisoma"]
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
        contact_id = [[0, i] for i in range(1, 19)],
        processor_num = unisoma_params["processor_num"],
        attention_heads_num = unisoma_params["attention_heads_num"],
        mlp_ratio = unisoma_params["mlp_ratio"],
        dropout = unisoma_params["dropout"],
        bias = unisoma_params["bias"],
        act = unisoma_params["act"]
    )

    if config["longtime"]["hyperparameter"]["criterion"] == "relative l2":
        criterion = relative_l2_loss
    elif config["longtime"]["hyperparameter"]["criterion"] == "mse l2":
        criterion = mse_l2_loss
    else:
        raise NotImplementedError

    normalizers = {
        "points": {},
        "edges": {}
    }

    for obj in ["load", "rigid", "init_deformable", "target_deformable"]:
        if obj in ["load", "rigid"]: 
            size = unisoma_params[f"{obj}_dim"]
            num = unisoma_params[f"{obj}_num"]
        elif obj == "init_deformable": 
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
        normalizers["edges"][f"init_deformable_{i}"] = Normalizer(
            size = 1,
            name = f"init_deformable_{i}_edges_normalizer",
            device = device
        )

    if run_type == "train":
        train(data_path, net, config, normalizers, criterion)
    elif run_type == "test":
        test(data_path, net, config, normalizers, criterion)


if __name__ == "__main__":
    config_path = "./configs/unilateral_stamping.toml"
    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = toml.load(f)

    version = config["longtime"]["version"]
    assets_root = config["longtime"]["assets_root"]
    data_root = config["longtime"]["data_root"]
    model_name = config["longtime"]["model_name"]

    device = config["longtime"]["hyperparameter"]["device"]
    log_path = f"{assets_root}/logs/{model_name}_longtime_log_{version}.log"
    log_writer = open(log_path, 'a', encoding = "utf8")

    sys.stdout = log_writer
    sys.stderr = log_writer

    print(config["longtime"]["description"])

    main("train")
    main("test")

    log_writer.close()

    # setsid python -u exp_unilateral_stamping_longtime.py &