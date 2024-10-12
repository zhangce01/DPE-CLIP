import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image
import json

import open_clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch
import torch.nn as nn

class TextEncoderWithPrompt(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, cupl_path, clip_model, coop=False, backbone='RN50'):
    if coop:
        n_ctx = 4
        if backbone == 'RN50':
            print('Using CoOp weights (RN50) for initialization.')
            coop_path = '/home/ce/DiffTPT/coop_weights/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
        elif backbone == 'ViT-B/16':
            print('Using CoOp weights (ViT-B/16) for initialization.')
            coop_path = '/home/ce/DiffTPT/coop_weights/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50'
        ctx = torch.load(coop_path)['state_dict']['ctx'].unsqueeze(0).cuda()
    f = open(cupl_path)
    cupl = json.load(f)
    
    if backbone == 'OpenCLIP':
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts += cupl[classname]
            
            if coop:
                prompts = [f'a photo of a {classname}.']
                tokenized_prompts = clip.tokenize(prompts).cuda()
                embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.visual.conv1.weight.dtype)

                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + n_ctx :, :]  # CLS, EOS
                
                # print(prefix.shape, ctx.shape, suffix.shape)

                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
                text_encoder_w_prompt = TextEncoderWithPrompt(clip_model)
                class_embedding = text_encoder_w_prompt(prompts, tokenized_prompts)
                class_embedding = class_embedding.squeeze()
            else:
                if backbone == 'RN50' or backbone == 'ViT-B/16':
                    texts = clip.tokenize(texts).cuda()
                elif backbone == 'OpenCLIP':
                    texts = tokenizer(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
                # prompt ensemble for ImageNet
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()           
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights):
    # with torch.no_grad():
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()
    
    # Change 3D tensor to 4D tensor
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    image_features = clip_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    clip_logits = 100. * image_features @ clip_weights

    if image_features.size(0) > 1:
        batch_entropy = softmax_entropy(clip_logits)
        selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
        output = clip_logits[selected_idx]
        image_features = image_features[selected_idx].mean(0).unsqueeze(0)
        clip_logits = output.mean(0).unsqueeze(0)

        loss = avg_entropy(output)
        prob_map = output.softmax(1).mean(0).unsqueeze(0)
        pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
    else:
        loss = softmax_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

    return image_features, clip_logits, loss, prob_map, pred


def get_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                             std=[0.5, 0.5, 0.5]) # For OpenCLIP
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        preprocess = get_preprocess()
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        # preprocess = get_preprocess()
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template, dataset.cupl_path