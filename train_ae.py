import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import MinkowskiEngine as ME

import torch.nn.functional as F 
from sparse_dataset import NearDetDataset3D
from completion_network import CompletionNet
import lightning as L
from lightning.fabric import Fabric

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=500)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="debug_completion.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--max_visualization", type=int, default=4)
parser.add_argument("--in_channel", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--feature_loss", type=bool, default=True)
parser.add_argument("--num_devices", type=int, default=1)

def collation_function(data):
    target_data = [datum['target'] for datum in data]
    input_data = [datum['input'] for datum in data]

    target_coords, target_feats = list(zip(*target_data))
    input_coords, input_feats = list(zip(*input_data))

    # Create batched coordinates for the SparseTensor input
    batched_target_coords = ME.utils.batched_coordinates(target_coords)
    batched_input_coords = ME.utils.batched_coordinates(input_coords)

    # Concatenate all lists
    batched_target_feats = torch.from_numpy(np.concatenate(target_feats, 0)).float()
    batched_input_feats = torch.from_numpy(np.concatenate(input_feats, 0)).float()

    return {
        'input': (batched_input_coords, batched_input_feats),
        'target': (batched_target_coords, batched_target_feats)
    }

def make_data_loader(
    dataset, batch_size, shuffle, collation_function, num_workers, config=None
):

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collation_function,
        "pin_memory": False,
        "drop_last": False,
    }

    args["shuffle"] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader

def calculate_loss(out_cls, targets, sout, sin, config, device):
    crit = nn.BCEWithLogitsLoss()

    num_layers, loss = len(out_cls), 0
    recon_f_loss = torch.tensor([0.0], device=device)

    recon_f_loss = torch.tensor([0.0], device=device)
    if config.feature_loss:
        batch_size = sin.C[:, 0].max().item()
        for batch_idx in range(batch_size):
            recon_mask = (sout.C[:, 0] == batch_idx)
            x_mask = (sin.C[:, 0] == batch_idx)
            if (recon_mask.sum() == 0) or (x_mask.sum() == 0) :
                continue
            coords_recon = sout.C[recon_mask, 1:] #.cpu()
            feats_recon = sout.F[recon_mask]
            coords_x = sin.C[x_mask, 1:] #.cpu()
            feats_x = sin.F[x_mask]
            # divide nearest neighbor computation in chuck of 1000
            sub_batch_size = 1000
            n_sub_batches = coords_recon.shape[0] // sub_batch_size + 1
            for sub_batch_idx in range(n_sub_batches):
                cur_inds = range(sub_batch_idx*sub_batch_size, min((sub_batch_idx+1)*sub_batch_size, coords_recon.shape[0]))
                if len(cur_inds) > 0:
                    cur_coords_recon = coords_recon[cur_inds]
                    cur_coords_recon = cur_coords_recon.reshape(-1, 1, cur_coords_recon.shape[-1])
                    closest_x_coodinates = (coords_x - cur_coords_recon).pow(2).sum(-1).min(-1).indices
                    recon_f_loss = F.mse_loss(feats_recon[cur_inds],
                                                        feats_x[closest_x_coodinates],
                                                        reduction='sum') / feats_x[closest_x_coodinates].size(0)
                    
    losses = []
    for out_cl, target in zip(out_cls, targets):
        curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
        losses.append(curr_loss.item())
        loss += curr_loss / num_layers
    # add features loss
    total_loss = loss + recon_f_loss
    return total_loss


def train(net, dataloader, device, config):
    # wrap with fabric for multi-gpu training
    fabric = Fabric()
    fabric.launch()
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    # setup optimizer and scheduler
    net, optimizer, scheduler = fabric.setup(net, optimizer, scheduler)
    dataloader = fabric.setup_dataloader(dataloader)
    
    net.train()
    logging.info(f"LR: {scheduler.get_lr()}")

    for epoch in range(config.num_epochs):
        for i, data_dict in enumerate(dataloader):
            optimizer.zero_grad()
            input_coords, input_features = data_dict['input']
            in_feat = torch.ones((len(input_coords), 1))

            sin = ME.SparseTensor(
                features=input_features,
                coordinates=input_coords,
                device=device,
            )
            
            target_coords, target_features = data_dict['target']

            # Generate target sparse tensor
            cm = sin.coordinate_manager
            target_key, _ = cm.insert_and_map(
                target_coords.to(device),
                string_id="target",
            )
            # Generate from a dense tensor
            out_cls, targets, _, sout = net(sin, target_key)
            num_layers, loss = len(out_cls), 0
            
            total_loss = calculate_loss(out_cls, targets, sout, sin, config, device)
            fabric.backward()
            optimizer.step()

            if i % config.stat_freq == 0:
                logging.info(
                    f"Iter: {i}, Loss: {loss.item():.3e}"
                )
            print(
                f"Iter: {i}, Total Loss: {total_loss.item():.3e}"
            )

            if i % config.val_freq == 0 and i > 0:
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "curr_iter": i,
                    },
                    config.weights,
                )

                scheduler.step()
                logging.info(f"LR: {scheduler.get_lr()}")

                net.train()


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NearDetDataset3D()

    dataloader = make_data_loader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collation_function=collation_function,
        num_workers=4
    )

    net = CompletionNet(config.resolution)
    net.to(device)

    logging.info(net)
    train(net, dataloader, device, config)
