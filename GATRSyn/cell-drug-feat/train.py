import argparse
import os
import os.path as osp
import random
import logging

from datetime import datetime
from time import perf_counter as t
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from const import COO_FILE, NODE_FEAT_FILE, DATA_DIR
from model import GATEncoder, Cell2Vec
from dataset import C2VDataset
from utils import save_args, save_best_model, save_and_visual_loss, find_best_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device)

edge_angle = pd.read_csv(r'normalized_edge_angle_loops.csv',header=None)
edge_angle_np = edge_angle.values
edge_angle = torch.tensor(edge_angle_np, dtype=torch.float32)

def train_step(mdl: Cell2Vec, edge_indices, node_x,edge_angle, node_idx, cell_idx, y_true, cell_sim, alpha):
    mdl.train()
    optimizer.zero_grad()
    y_pred, attention ,cell_cell = mdl(edge_indices, node_x, edge_angle, node_idx,cell_idx)
    loss = loss_func(y_pred, y_true)
    loss_cell = loss_func(cell_cell, cell_sim)
    step_loss = (1 - alpha) * loss + alpha * (loss_cell)
    step_loss.backward()
    optimizer.step()
    return step_loss.item(), attention

def gen_emb(mdl: Cell2Vec):
    mdl.eval()
    with torch.no_grad():
        emb = mdl.embeddings.weight.data
    return emb.cpu().numpy()

def p_type(x):
    if isinstance(x, list):
        for xx in x:
            assert 0 <= xx < 1
    else:
        assert 0 <= x < 1
    return x


def get_graph_data():
    edges = np.load(COO_FILE).astype(int).transpose()
    eid = torch.from_numpy(edges)
    feat = np.load(NODE_FEAT_FILE)
    feat = torch.from_numpy(feat).float()
    return eid, feat


if __name__ == '__main__':
    start_time = time.time()

    time_str = str(datetime.now().strftime('%y%m%d%H%M'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='target_mut.npy', help="target feature")
    parser.add_argument('--valid_nodes', type=str, default='nodes_mut.npy', help="list of valid nodes")
    parser.add_argument('--cell_sim', type=str, default='cell_mut_sim.npy', help="cell/drug similarty")
    parser.add_argument('--conv', type=int, default=2,
                        help="the number of graph conv layer, must be no less than 2")
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="dim of hidden space for GCN layer")
    parser.add_argument('--loss', type=str, choices=['mse', 'bce'], default='mse')
    parser.add_argument('--emb_dim', type=int, default=384, help="dim of cell/drug embeddings:384,128")
    parser.add_argument('--active', type=str, choices=['relu', 'prelu', 'elu', 'leaky'], default='relu',
                        help="activate function")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--epoch', type=int, default=1000, help="number of epochs to train.")
    parser.add_argument('--batch', type=int, default=2, help="batch size")
    parser.add_argument('--keep', type=int, default=1, help="max number of best models to keep")
    parser.add_argument('--patience', type=int, default=50, help="patience")
    parser.add_argument('--loss_type', type=str, default="PRELU", help="RELU/Leaky/PRELU")
    parser.add_argument('--simple_distance', type=str, default="N", help="Y/N; Whether multiplying or embedding positional information")
    parser.add_argument('--with_edge', type=str, default="Y", help="Y/N")
    parser.add_argument('--norm_type', type=str, default="layer", help="BatchNorm=batch/LayerNorm=layer")
    parser.add_argument('--gpu', type=int, default=0,
                        help="gpu id to use")
    parser.add_argument('--suffix', type=str, default=time_str, help="suffix for model dir")
    args = parser.parse_args()
    print(args)
    t_type = args.target.split('.')[0].split('_')[1]
    mdl_dir = osp.join(DATA_DIR, 'OnePPI_sim_0.2_{}_({}x2)x{}_{}'.format(t_type, args.hidden_dim, args.emb_dim, args.suffix))
    loss_file = osp.join(mdl_dir, 'loss.pkl')
    if not osp.exists(mdl_dir):
        os.makedirs(mdl_dir)
    save_args(args, osp.join(mdl_dir, 'args384.json'))
    log_file = os.path.join(mdl_dir, 'train.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)

    torch.manual_seed(23333)
    random.seed(12345)

    learning_rate = args.lr
    hidden_dim = args.hidden_dim
    emb_dim = args.emb_dim
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'elu': nn.ELU(), 'leaky': nn.LeakyReLU()})[args.active]
    num_layers = args.conv
    batch_size = args.batch
    num_epochs = args.epoch
    weight_decay = 1e-5
    loss_type = ({'RELU': F.relu, 'PRELU': nn.PReLU(), 'Leaky': nn.LeakyReLU()})[args.loss_type]
    simple_distance = args.simple_distance
    norm_type = args.norm_type
    with_edge = args.with_edge

    edge_indices, node_features = get_graph_data()
    edge_angle = edge_angle.to(device)
    num_nodes = torch.max(edge_indices) + 1
    self_loops = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0)
    self_loops = self_loops.repeat(2, 1)
    edge_indices_with_loops = torch.cat((edge_indices, self_loops), dim=1)
    edge_indices = edge_indices_with_loops.to(device)

    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    node_features = node_features.to(device)
    c2v_dataset = C2VDataset(osp.join(DATA_DIR, args.target),
                             osp.join(DATA_DIR, args.valid_nodes),
                             osp.join(DATA_DIR, args.cell_sim))
    dataloader = DataLoader(c2v_dataset, shuffle=True, num_workers=2)
    node_indices = c2v_dataset.node_indices.to(device)
    encoder = GATEncoder(node_features.shape[1], hidden_dim,2,0.5,0.5,loss_type,with_edge,simple_distance,norm_type).to(device)
    model = Cell2Vec(encoder, len(c2v_dataset), emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if args.loss == 'mse':
        loss_func = nn.MSELoss(reduction='mean')
    else:
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    logging.info("Check model.")
    logging.info(model)

    logging.info("Start training.")
    losses = []
    min_loss = float('inf')
    angry = 0

    start = t()
    prev = start
    best_model_info = {
        'model_state_dict': None,
        'attention': None,
        'epoch': None
    }
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        now = t()
        for step, batch in enumerate(dataloader):
            batch_x, batch_y, batch_z = batch
            print(batch)
            batch_x, batch_y, batch_z = batch_x.to(device), batch_y.to(device), batch_z.to(device)
            loss, attention = train_step(model, edge_indices, node_features, edge_angle, node_indices, batch_x, batch_y,batch_z, 0.2)
            epoch_loss += loss * len(batch_x)
        epoch_loss /= len(c2v_dataset)
        logging.info('Epoch={:04d} Loss={:.4f}'.format(epoch, epoch_loss))
        losses.append(epoch_loss)
        prev = now
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model_info['model_state_dict'] = model.state_dict()
            best_model_info['attention'] = attention
            best_model_info['epoch'] = epoch
            save_best_model(best_model_info, mdl_dir, epoch, args.keep)
            angry = 0
        else:
            angry += 1
        if angry == args.patience:
            break

    logging.info("Training completed.")
    logging.info("Min train loss: {:.4f} | Epoch: {:04d}".format(min_loss, losses.index(min_loss)))
    logging.info("Save to {}".format(mdl_dir))

    save_and_visual_loss(losses, loss_file, title='Train Loss', xlabel='epoch', ylabel='Loss')
    logging.info("Save train loss curve to {}".format(loss_file))

    checkpoint = torch.load(find_best_model(mdl_dir), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    embeddings = gen_emb(model)
    np.save(os.path.join(mdl_dir, 'embeddings384.npy'), embeddings)
    logging.info("Save {}".format(
        os.path.join(mdl_dir, 'embeddings384.npy')))

    best_attention = best_model_info['attention']
    best_attention_cpu = best_attention.detach().cpu().numpy()
    np.save(os.path.join(mdl_dir, 'best_attention.npy'), best_attention_cpu)
    logging.info("Save {}".format(os.path.join(mdl_dir, 'best_attention.npy')))

    end_time = time.time()
    run_time = end_time - start_time

    hours = int(run_time // 3600)
    minutes = int((run_time % 3600) // 60)
    seconds = int(run_time % 60)
    print(f"运行时间：{hours}小时 {minutes}分钟 {seconds}秒钟")
