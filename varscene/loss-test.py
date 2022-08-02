import pickle
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os
import torch
import argparse
import networkx as nx
import importlib
# from torchinfo import summary
from configure import get_default_config
from utils import build_model, get_graph, get_star_embedding, pack_batch, batch_sample_graphs, get_node_star, get_star_dict, get_star_embedding, batch_sample_graphs_from_z, \
    get_label_embeddings, get_star_dict, pack_batch

from collections import Counter
from calc_metrics import graph_structure_worker
from calc_metrics import get_node_bigrams as get_node_bi, get_edge_bigrams as get_edge_bi, make_dist_mmd_opt
from calc_metrics import cos_sim, get_all_stars as get_star_sim, compute_mmd, single_star_worker_mmd_opt, gaussian


total_run_time = time.time()

out_dir = './0726'
dataset = 'svg'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
os.system('cp configure.py %s' % (os.path.join(out_dir, 'configure.py')))

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

t1 = time.time()
## load graphs
if dataset == 'vg': ## visual genome
    train_data_dir = '../vg_data/data'
elif dataset == 'vrd': ## visual relationship detection
    train_data_dir = '../vrd_data/data'
elif dataset == 'svg': ## small-sized visual genome
    train_data_dir = '../svg_data/data'
else:
    raise ValueError('invalid dataset %s, exiting...' % dataset)

with open(os.path.join(train_data_dir, 'graphs_train.pkl'), 'rb') as f:
    training_set = pickle.load(f)
with open(os.path.join(train_data_dir, 'graphs_val.pkl'), 'rb') as f:
    validation_set = pickle.load(f)

training_set = [nx.convert_node_labels_to_integers(g) for g in training_set]
validation_set = [nx.convert_node_labels_to_integers(g) for g in validation_set]


print('loaded graphs in %.2fs' % (time.time()-t1))

t1 = time.time()
node_label_embeddings, edge_label_embeddings = get_label_embeddings(training_set)
training_star_dict = get_star_dict(training_set)

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

training_star_embeddings = get_star_embedding(training_star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)

print('computed stars and embeddings in %.2fs' % (time.time()-t1))

## initialize model and optimizer
model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

train_epochs, train_loss_hist = [], []
start_epoch = 0

model_file = os.path.join(out_dir, ('star_vae_%s.pt' % dataset))
optimizer_file = os.path.join(out_dir, ('optimizer_%s.pt' % dataset))
loss_hist_file = os.path.join(out_dir, 'loss_hist.pkl')

batch_size = 1024
n_training_steps = 1

model_file = os.path.join(out_dir, ('star_vae_%s_%s.pt' % (dataset, start_epoch+n_training_steps)))
optimizer_file = os.path.join(out_dir, ('optimizer_%s_%s.pt' % (dataset, start_epoch+n_training_steps)))
loss_hist_file = os.path.join(out_dir, 'loss_hist_%s.pkl' % (start_epoch+n_training_steps))

## prepare batches
t1 = time.time()

print('starting training...')
for i_iter in range(start_epoch, start_epoch+n_training_steps):
    train_running_loss = 0
    t1 = time.time()

    # for batch in batch_data:
    for i_batch in range(0, len(training_set), batch_size):
        batch = pack_batch(training_set[i_batch : i_batch+batch_size],
                        node_label_embeddings, edge_label_embeddings,
                        training_star_embeddings, training_star_dict)
    
        model.train(mode=True)
        optimizer.zero_grad()
        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        # t2 = time.time()
        loss = model(node_features.to(device), edge_features.to(device), star_features.to(device),
            candidate_star_features.to(device), target_star_idx.to(device),
            star_z_mask.to(device), candidate_star_z_mask.to(device),
            from_idx.to(device), to_idx.to(device), graph_idx.to(device), batch_size, 0.1,
            graph_depth_range.to(device), node_graph_depth_idx.to(device), False)

        loss.backward()
        optimizer.step()
        # print(optimizer)
        train_running_loss += loss.item()
        # print('%.4fs for batch update' % (time.time()-t2))
        
    train_running_loss /= len(training_set)
    train_loss_hist.append(train_running_loss)
    train_epochs.append(i_iter+1)

    ## save model
    if i_iter % config['training']['save_model_after'] == 0 or i_iter == n_training_steps-1+start_epoch:
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
    
    print('epoch %s, train loss %.6f, time %.2fs' %(i_iter+1, train_running_loss, time.time()-t1))

    if i_iter % config['training']['save_loss_hist_after'] == 0 or \
        i_iter+1 == n_training_steps+start_epoch:
        with open(loss_hist_file, 'wb') as f:
            pickle.dump((train_epochs, train_loss_hist), f)
        
        plt.figure()
        plt.plot(train_epochs, train_loss_hist, label='train loss')
        plt.legend()
        plt.title('Loss history')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(loss_hist_file.replace('pkl', 'png'), bbox_inches='tight')
        plt.close()

print('total run time %.2fm' % ((time.time()-total_run_time)/60))

config = importlib.import_module('0726.configure').get_default_config()
# Set random seeds
seed = config['seed']
random.seed(seed + 2)
np.random.seed(seed + 3)
torch.manual_seed(seed + 4)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

t1 = time.time()
node_label_embeddings, edge_label_embeddings = get_label_embeddings(training_set+validation_set)
print('loaded sentence embedding in %.2fs' % (time.time()-t1))

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

star_dict = get_star_dict(training_set + validation_set)

node_feature_dim = list(node_label_embeddings.values())[0].shape[-1]
edge_feature_dim = list(edge_label_embeddings.values())[0].shape[-1]

star_embeddings = get_star_embedding(star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)
## initialize model and optimizer
model, _ = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

with open(os.path.join(train_data_dir, 'all_stars.pkl'), 'rb') as f:
    all_stars = pickle.load(f)
with open(os.path.join(train_data_dir, 'all_pair_stars.pkl'), 'rb') as f:
    all_pairs = pickle.load(f)

validation_star_dict = get_star_dict(validation_set)
validation_star_embeddings = get_star_embedding(validation_star_dict.keys(), node_label_embeddings,
                                    edge_label_embeddings, edge_feature_dim)


## compute ids of stars for MMD

mmd_choices = ['node_edge', 'star_node_edge', 'sp_node_edge', 'sp', 'wl', 'nspd', 'node_bi', 'edge_bi', 'star_sim', 'star']
mmd = 'star'

t1 = time.time()
sample_star_for_MMD = 1000
idx2sampled_stars = [star for star, _ in Counter(all_stars).most_common(sample_star_for_MMD)]
sampled_stars2idx = {}
for i, e in enumerate(idx2sampled_stars): sampled_stars2idx[e] = i

star_embeddings = training_star_embeddings.copy()
star_embeddings.update(validation_star_embeddings)
star_dict = training_star_dict.copy()
star_dict.update(validation_star_dict)

print('computed stars and embeddings in %.2fs' % (time.time()-t1))

if mmd == 'star':
    mmd_kernel = single_star_worker_mmd_opt
    mmd_kernel_kwargs = dict(sampled_stars2idx=sampled_stars2idx)
elif mmd == 'sp' or args.mmd == 'sp_node_edge':
    grkl_kernel = ShortestPath(normalize=False)
elif mmd == 'nspd':
    grkl_kernel = NeighborhoodSubgraphPairwiseDistance(normalize=False)
elif mmd == 'wl':
    grkl_kernel = WeisfeilerLehman(normalize=False)

def structure_cos_sim(gphs1, gphs2, worker):
    all_strucs = list()
    for g in gphs1+gphs2:
        all_strucs.extend(worker(g))
    all_strucs = list(set(all_strucs))
    idx2strucs = all_strucs
    struc2idx = {}
    for i, e in enumerate(idx2strucs): struc2idx[e] = i
    dist1 = graph_structure_worker(gphs1, struc2idx, worker)
    dist2 = graph_structure_worker(gphs2, struc2idx, worker)
    return cos_sim(dist1, dist2)
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    

def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

import json
sg2im_checkpoint = torch.load('../sg2im/sg2im-models/vg128.pt', map_location='cpu')
vocab = sg2im_checkpoint['model_kwargs']['vocab']
object_vocab = vocab['object_name_to_idx']
pred_vocab = vocab['pred_name_to_idx']

import sys
sys.path.insert(1, '../sg2im')
from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.discriminators import PatchDiscriminator
from sg2im.losses import get_gan_losses
sg2im_model = Sg2ImModel(**sg2im_checkpoint['model_kwargs'])
sg2im_model.load_state_dict(sg2im_checkpoint['model_state'])
sg2im_model.eval()
sg2im_model.to(device)

from torch.utils.data import DataLoader
## initialize model
optim_model, _ = build_model(config, node_feature_dim, edge_feature_dim)
## set optimizer only for decoder
optimizer = torch.optim.Adam((optim_model._decoder.parameters()),
        lr=1e-3, weight_decay=1e-5)
optim_model.to(device)

lr=1e-3
batch_size = 1024
n_training_steps = 10
kl_weight = 1000
num_graphs_mmd = 1000
kl_graphs_batch_size = 1024

train_epochs, train_loss_hist = list(), list()
train_mmd_hist = list()
train_mmd_loss_component_hist = list()
train_kl_loss_component_hist = list()
start_epoch = 0

model_file = os.path.join(out_dir, 'mmd_log_model_%s_%s_%s.pt' % (kl_weight, mmd, lr))
optimizer_file = os.path.join(out_dir, 'mmd_log_model_optimizer_%s_%s_%s.pt' % (kl_weight, mmd, lr))
loss_hist_file = os.path.join(out_dir, 'mmd_log_loss_hist_%s_%s_%s.pkl' % (kl_weight, mmd, lr))
loss_hist_plots_dir = os.path.join(out_dir, 'mmd_log_model_%s_%s_%s_loss_plots' % (kl_weight, mmd, lr))
if not os.path.isdir(loss_hist_plots_dir):
    os.mkdir(loss_hist_plots_dir)


def nx_to_json(graphs):
    sg_list = list()
    for g in graphs:
        nodes = g.nodes(data='label')
        objects = [label for _, label in nodes]
        relationships = list()
        for u, v, label in g.edges(data='label'):
            relationships.append([u, label, v])
        sg_list.append(dict(objects=objects, relationships=relationships))
    return sg_list

def graph_in_vocab(g, object_vocab, pred_vocab):
    for _, l in g.nodes(data='label'):
        if l not in object_vocab:
            return False
    for _, _, l in g.edges(data='label'):
        if l not in pred_vocab:
            return False
    return True

def build_img_discriminator(vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = 1.0
    d_img_weight = 1.0
    if d_weight == 0 or d_img_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'arch': 'C4-64-2,C4-128-2,C4-256-2',
        'normalization': 'batch',
        'activation': 'leakyrelu-0.2',
        'padding': 'valid',
    }
    discriminator = PatchDiscriminator(**d_kwargs)
    return discriminator, d_kwargs


print('starting training...')
for i_iter in range(start_epoch, start_epoch+n_training_steps):

    t1 = time.time()
    optimizer.zero_grad()
    loss = 0
    kl_loss_component = 0
    mmd_loss_component = 0
    model.eval()
    # summary(model, input_size=(batch_size, 1, 28, 28))
    total_time = 0 

    ## compute kl-div loss over entire training graphs
    kl_graphs_num = i_iter % max(1, (len(training_set)//kl_graphs_batch_size))
    kl_graphs = training_set[kl_graphs_num*kl_graphs_batch_size : (kl_graphs_num+1)*kl_graphs_batch_size]
    # for i_batch in range(0, len(kl_graphs), batch_size):
    for i_batch in range(0, 1):
        train_batch = kl_graphs[i_batch : i_batch + batch_size]

        ## sample z values and first stars from generation graphs
        batch = pack_batch(train_batch,
                        node_label_embeddings, edge_label_embeddings,
                        star_embeddings, star_dict)
        first_star_list = batch.graph_first_star

        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        node_features = node_features.to(device)
        edge_features = edge_features.to(device)
        star_features = star_features.to(device)
        candidate_star_features = candidate_star_features.to(device)
        target_star_idx = target_star_idx.to(device)
        star_z_mask = star_z_mask.to(device)
        candidate_star_z_mask = candidate_star_z_mask.to(device)
        from_idx = from_idx.to(device)
        to_idx = to_idx.to(device)
        graph_idx = graph_idx.to(device)
        edge_graph_idx = edge_graph_idx.to(device)
        graph_depth_range = graph_depth_range.to(device)
        node_graph_depth_idx = node_graph_depth_idx.to(device)

        tg = time.time()
        with torch.no_grad():
            _, z_list = model(node_features,
                edge_features, star_features,
                candidate_star_features, target_star_idx,
                star_z_mask, candidate_star_z_mask,
                from_idx, to_idx, graph_idx, 1+int(torch.max(graph_idx)),
                0, graph_depth_range, node_graph_depth_idx, True)
        ## for each graph, obtained z values in `z_list` and first star in `first_star_list`

        log_prob_list = optim_model.log_prob_given_z(z_list, star_features,
                                candidate_star_features, target_star_idx,
                                star_z_mask, candidate_star_z_mask,
                                to_idx, edge_graph_idx, 1+int(torch.max(graph_idx)),
                                graph_depth_range)

        with torch.no_grad():
            base_log_prob_list = model.log_prob_given_z(z_list, star_features,
                                        candidate_star_features, target_star_idx,
                                        star_z_mask, candidate_star_z_mask,
                                        to_idx, edge_graph_idx, 1+int(torch.max(graph_idx)),
                                        graph_depth_range)

        loss = (kl_weight/2*torch.sum((log_prob_list-base_log_prob_list)**2)\
            +kl_weight*torch.sum(log_prob_list)) / len(kl_graphs)
        kl_loss_component += loss.item()
        loss.backward()
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&--optim--&&&&&&&&&&&&&&&&&&&&&&&&&')
        # dir(model)
        # print(dir(model))
        # print('&&&&&&&&&&&&&&&&&&&&&&&&&--sg2im--&&&&&&&&&&&&&&&&&&&&&&&&&')
        # dir(sg2im_model)
        # print(dir(sg2im_model))
        # # plot_grad_flow(model.named_parameters())

        total_time += time.time()-tg
    

    ## mmd loss computation
    gen_graphs_num = i_iter % (len(training_set)//num_graphs_mmd)
    generation_graphs = training_set[gen_graphs_num*num_graphs_mmd : (gen_graphs_num+1)*num_graphs_mmd]
    mmd_graphs = random.sample(validation_set, min(len(validation_set), num_graphs_mmd))
    first_star_list, z_list = list(), list()

    ## sample z values and first stars from generation graphs
    for i_batch in range(0, len(generation_graphs), batch_size):
        gen_graph_batch = generation_graphs[i_batch : i_batch+batch_size]

        batch = pack_batch(gen_graph_batch,
                        node_label_embeddings, edge_label_embeddings,
                        star_embeddings, star_dict)
        first_star_list.extend(batch.graph_first_star)

        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, _, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        tg = time.time()
        with torch.no_grad():
            _, batch_z_list = model(node_features.to(device),
                edge_features.to(device), star_features.to(device),
                candidate_star_features.to(device), target_star_idx.to(device),
                star_z_mask.to(device), candidate_star_z_mask.to(device),
                from_idx.to(device), to_idx.to(device), graph_idx.to(device), 1+int(torch.max(graph_idx)),
                0, graph_depth_range.to(device), node_graph_depth_idx.to(device), True)
        total_time += time.time()-tg
        z_list.extend(batch_z_list)
    ## for each graph, obtained z values in `z_list` and first star in `first_star_list`

    ## sample graphs conditioned on `z_list`
    t2 = time.time()
    tg = time.time()
    new_graphs, utilized_z_list, _, _ = batch_sample_graphs_from_z(optim_model, z_list, first_star_list,
                                    cutoff_size=20, n_trials=50, n_sampling_epochs=1,
                                    verbose=False, star_dict=star_dict, star_embeddings=star_embeddings)
    print('sampled %s/%s graphs in %.2fs' % (len(new_graphs), len(generation_graphs), time.time()-t2))
    total_time += time.time()-tg
    sg_list = list()
    for g in new_graphs:               
        if graph_in_vocab(g, object_vocab, pred_vocab):
            gph_set = [nx.convert_node_labels_to_integers(nx.DiGraph(g))]
            sg_list.extend(nx_to_json(gph_set))

    img_total_loss = 0
    for i_test in range(0,512):
        objs, triples, obj_to_img  = sg2im_model.encode_scene_graphs(sg_list[i_test : 2+i_test])
        imgs_pred, boxes_pred, masks_pred, predicate_scores = sg2im_model.forward(objs, triples, obj_to_img)
        gan_g_loss, gan_d_loss = get_gan_losses('gan')
        img_discriminator, d_img_kwargs = build_img_discriminator(vocab)
        imgs_pred = imgs_pred.to(torch.device('cuda'))
        img_discriminator = img_discriminator.to(torch.device('cuda'))
        scores_fake = img_discriminator(imgs_pred)
        img_loss = gan_g_loss(scores_fake)
        img_total_loss += img_loss
        
    # img_total_loss.backward()
    
    # ## plot loss history
    # if i_iter % config['training']['save_loss_hist_after'] == 0 or i_iter == n_training_steps-1+start_epoch:
    #     plt.figure()
    #     plt.plot(train_epochs, eval('train_loss_hist'), label='train loss')
    #     plt.legend()
    #     plt.title('img recon loss')
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.savefig(os.path.join(loss_hist_plots_dir, 'loss_hist.png'), bbox_inches='tight')
    #     plt.close()

    ## compute MMD
    t2 = time.time()
    if mmd == 'star':
        dist_1 = make_dist_mmd_opt(mmd_graphs, mmd_kernel, **mmd_kernel_kwargs)
        dist_2 = make_dist_mmd_opt(new_graphs, mmd_kernel, **mmd_kernel_kwargs)
        star_mmd = compute_mmd(dist_1, dist_2, kernel=gaussian, sigma=1)
    elif mmd in ['star_sim', 'node_bi', 'edge_bi']:
        ## take negative since star_mmd is minimized
        star_mmd = -structure_cos_sim(new_graphs, mmd_graphs, eval('get_%s' % mmd))
    elif mmd in ['sp', 'wl', 'nspd']:
        ref_gphs = grakel.utils.graph_from_networkx(mmd_graphs, 'label', 'label')
        pred_gphs = grakel.utils.graph_from_networkx([nx.Graph(o) for o in new_graphs], 'label', 'label')
        grkl_kernel.fit_transform(ref_gphs)
        K_pred = grkl_kernel.transform(pred_gphs)
        K_pred = np.nan_to_num(K_pred)
        star_mmd = -np.mean(K_pred) ## take negative since star_mmd is minimized
    elif 'node_edge' in mmd:
        ## take negative since star_mmd is minimized
        star_mmd = -structure_cos_sim(new_graphs, mmd_graphs, get_node_bi)
        star_mmd += -structure_cos_sim(new_graphs, mmd_graphs, get_edge_bi)
        if mmd == 'star_node_edge':
            star_mmd += -structure_cos_sim(new_graphs, mmd_graphs, get_star_sim)
        elif mmd == 'sp_node_edge':
            ref_gphs = grakel.utils.graph_from_networkx(mmd_graphs, 'label', 'label')
            pred_gphs = grakel.utils.graph_from_networkx([nx.Graph(o) for o in new_graphs], 'label', 'label')
            grkl_kernel.fit_transform(ref_gphs)
            K_pred = grkl_kernel.transform(pred_gphs)
            K_pred = np.nan_to_num(K_pred)
            star_mmd += -np.mean(K_pred) ## take negative since star_mmd is minimized

    print('%s mmd %.6f computed in %.2fs' % (mmd, star_mmd, time.time()-t2))

    ## probability of `new_graphs` given their z-representations `utilized_z_list`
    for i_batch in range(0, len(new_graphs), batch_size):
        new_graphs_batch = new_graphs[i_batch : i_batch + batch_size]

        batch = pack_batch(new_graphs_batch,
                    node_label_embeddings, edge_label_embeddings,
                    star_embeddings, star_dict)
        node_features, edge_features, star_features, candidate_star_features,\
        target_star_idx, star_z_mask, candidate_star_z_mask, from_idx,\
        to_idx, graph_idx, edge_graph_idx, graph_depth_range, node_graph_depth_idx = get_graph(batch)

        tg = time.time()
        log_prob_list = model.log_prob_given_z(
                                [o.to(device) for o in utilized_z_list], star_features.to(device),
                                candidate_star_features.to(device), target_star_idx.to(device),
                                star_z_mask.to(device), candidate_star_z_mask.to(device),
                                to_idx.to(device), edge_graph_idx.to(device), 1+int(torch.max(graph_idx)),
                                graph_depth_range.to(device))

        loss = torch.sum(log_prob_list)*star_mmd
        mmd_loss_component += loss.item()
        loss.backward()
        total_time += time.time()-tg

    optimizer.step()
    train_loss_hist.append(mmd_loss_component+kl_loss_component)
    train_mmd_hist.append(star_mmd)
    train_mmd_loss_component_hist.append(mmd_loss_component)
    train_kl_loss_component_hist.append(kl_loss_component)
    train_epochs.append(i_iter+1)

    ## save model
    if i_iter % config['training']['save_model_after'] == 0 or i_iter == n_training_steps-1+start_epoch:
        torch.save(optim_model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        with open(os.path.join(out_dir, 'mmd_log_generated_%s_%s.pkl' % (kl_weight, mmd)), 'wb') as f:
            pickle.dump(new_graphs, f)

    print('epoch %s, loss %.4f, %s_mmd %.6f, mmd_loss %.4f, kl_loss %.4f, time %.2fs' %\
        (i_iter+1, mmd_loss_component+kl_loss_component, mmd, star_mmd, mmd_loss_component, kl_loss_component, time.time()-t1))


