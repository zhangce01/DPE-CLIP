import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn.functional as F
import operator
import torch.nn as nn
from info_nce import InfoNCE
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

import clip
from utils import *

import open_clip


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of DPE on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='../data/', help='Path to the datasets directory. Default is ../data/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16', 'SigLIP', 'OpenCLIP'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--coop', dest='coop', action='store_true', help='Whether you want to use CoOp weights for initialization.')

    args = parser.parse_args()

    return args

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
        return
    
def visualize_cache(cache, iter):
    # t-SNE visualization of cache features
    with torch.no_grad():
        cache_features = []
        cache_labels = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_features.append(item[0].reshape(-1))
                cache_labels.append(class_index)
        cache_features = torch.stack(cache_features, dim=0)
        cache_labels = torch.Tensor(cache_labels).to(torch.int64)
        cache_features = F.normalize(cache_features, dim=1)
        cache_features = cache_features.cpu().numpy()
        cache_labels = cache_labels.cpu().numpy()
        tsne = TSNE(n_components=2)
        print(cache_features.shape)
        cache_features_fit = tsne.fit_transform(cache_features)
        
        # Assign different colors to different cache_labels
        colors = [
            '#00429d',  # Strong Blue
            '#93003a',  # Deep Red
            '#007d34',  # Vivid Green
            '#ff6800',  # Vivid Orange
            '#e30022',  # Bright Red
            '#a6bdd7',  # Light Periwinkle
            '#ffcc00',  # Vivid Yellow
            '#540d6e',  # Dark Violet
            '#7f180d',  # Dark Red
            '#00939c',  # Cyan Process
            '#5f3c99',  # Purplish Blue
            '#ff4a46',  # Bright Red-Orange
            '#8f0075',  # Strong Purple
            '#ff3c38',  # Bright Red
            '#83a697',  # Muted Cyan
            '#1e96be',  # Strong Cyan
            '#d9e021',  # Vivid Lime Green
            '#f18d05',  # Rich Orange
            '#f6e120',  # Bright Yellow
            '#8f2d56',  # Strong Rose
            '#006837',  # Dark Green
            '#e7298a',  # Bright Pink
            '#ce1256',  # Dark Pink
            '#01665e',  # Dark Teal
            '#dfc27d',  # Pale Gold
            '#35978f',  # Muted Teal
            '#bf812d',  # Mustard Brown
            '#543005',  # Dark Brown
            '#8c510a',  # Light Brown
            '#80cdc1',  # Soft Turquoise
        ]
        colors_others = 'gray'
        figure, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 5))
        patch = ax.patch
        patch.set_color("#f5f5f5")
        ax.tick_params(axis='both',          # Changes apply to both x and y axes
               which='both',         # Apply changes to both major and minor ticks
               bottom=False,         # No ticks along the bottom edge
               top=False,            # No ticks along the top edge
               left=False,           # No ticks along the left edge
               right=False,          # No ticks along the right edge
               labelbottom=False,    # No labels along the bottom edge
               labelleft=False)      # No labels along the left edge
        plt.grid(color='w', zorder=0, linewidth=2)
        plt.gca().spines['bottom'].set_color('gray')
        plt.gca().spines['left'].set_color('gray')
        plt.gca().spines['top'].set_color('gray')
        plt.gca().spines['right'].set_color('gray')
        # In Food-101, we have 101 classes
        for i in range(101):
            if i < 30:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], c=colors[i], s=15, marker='x', zorder=5)
            else:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], c=colors_others, s=5, zorder=5)
        save_path = 'fig/cache_features_iter_{}.png'.format(iter)
        plt.savefig(save_path)
        plt.close()
        

def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the prototype of the class
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
            
        return cache_keys, cache_values, all_classes
    
def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits
    
class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights
    
    def reset(self):
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def run_test_dpe(pos_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name):   
    with torch.cuda.amp.autocast():
        pos_cache, accuracies = {}, []
        
        # Unpack all hyperparameters
        pos_enabled = pos_cfg['enabled']
        
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        
        clip_weights_global = clip_weights.clone()
        num_avg = 0
        total = len(loader)
        
        losses = []
        all_clip_weights = []
        distances = []

        # Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local)
            new_clip_weights = text_residue(clip_weights_local)

            image_features_x, clip_logits, entropy, prob_map, pred = get_clip_logits(images, clip_model, new_clip_weights)
            target = target.cuda()
            
            if pos_enabled:
                entropy = get_entropy(entropy, clip_weights)
                update_cache(pos_cache, pred, [image_features_x, entropy], pos_params['shot_capacity'])
                pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
                # if i != 0 and i % 1000 == 0:
                #     visualize_cache(pos_cache, i)
            steps = 1 # Update step, set to 1 in default
            for j in range(steps):
                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(image_features_x, new_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)
                    loss = avg_entropy(final_logits)
                    # alignment loss
                    image2text_loss = InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T)
                    loss += image2text_loss * lr_cfg['align']
                else:
                    loss = avg_entropy(final_logits)
                
                lr_text = lr_cfg['text']
                lr_image = lr_cfg['image']
                if pos_enabled and pos_cache:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1},
                        {'params': pos_cache_residue.parameters(), 'lr': lr_image, 'eps': 1e-3, 'weight_decay': 1e-1}
                        ])
                else:
                    optimizer = torch.optim.AdamW([
                        {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}
                        ])

                optimizer.zero_grad()
                if j == steps - 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                optimizer.step()

            # Actual inference
            text_residue.eval()
            if pos_enabled and pos_cache:
                pos_cache_residue.eval()
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                if dataset_name == 'A':
                    image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, new_clip_weights)
                else:
                    image_features, clip_logits, _, _, _ = get_clip_logits(images[0], clip_model, new_clip_weights)
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(image_features, new_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)       
                    
                acc = cls_acc(final_logits, target.cuda())  
                accuracies.append(acc)
                wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
                
                loss = avg_entropy(final_logits)
                
                # Global update step: textual prototype evolution
                # lam = 0.99
                # clip_weights_global = sum([w * clip for w, clip in zip(weights, all_clip_weights)])
                if get_entropy(loss, clip_weights) < 0.1:
                    # Full Update
                    # clip_weights_global = new_clip_weights 
                    # Cumalative Avg
                    num_avg += 1
                    clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + new_clip_weights * (1 / (num_avg + 1))
                    # clip_weights_global = clip_weights_global / clip_weights_global.norm(dim=0)
                    # Exponential Avg
                    # clip_weights_global = clip_weights_global * lam + new_clip_weights * (1 - lam)
        
            if i % 1000 == 0:
                print("---- DPE's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
    print("---- DPE's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
    

    return sum(accuracies)/len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    if args.backbone == 'RN50' or args.backbone == 'ViT-B/16':
        clip_model, preprocess = clip.load(args.backbone)
    elif args.backbone == 'SigLIP':
        clip_model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
    elif args.backbone == 'OpenCLIP':
        clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        clip_model = clip_model.to('cuda')

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run DPE on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        # Set random seed
        random.seed(1)
        torch.manual_seed(1)
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        print(args.coop)
        print(args.backbone)
        
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, cupl_path, clip_model, args.coop, args.backbone)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc = run_test_dpe(cfg['positive'], cfg['learning_rate'], test_loader, clip_model, clip_weights, dataset_name)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()