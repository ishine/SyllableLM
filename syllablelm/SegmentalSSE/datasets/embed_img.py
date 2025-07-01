import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import tqdm
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import json
import os
import numpy as np
import torch

json_fns = ["/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_test_karpathy.json", "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_val_karpathy.json", "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_train_karpathy.json"]
# json_fns = ["/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_test_karpathy.json"]

img_root = "/data/scratch/pyp/datasets/coco_pyp/coco_img"

save_root = "/data/scratch/pyp/datasets/coco_pyp/preextract_embedding_clip_img"

os.makedirs(save_root, exist_ok=True)

clip_version = "ViT-L/14@336px" 

clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768, "ViT-L/14@336px": 768}[clip_version]
clip_img_res = {'ViT-L/14': 224, "ViT-L/14@336px": 336}[clip_version]

clip_model, _ = clip.load(clip_version)  # clip.available_models()
preprocess = Compose([
        Resize(clip_img_res, interpolation=BICUBIC),
        CenterCrop(clip_img_res),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
clip_model.cuda().eval()

def clip_embed_img(img):
    assert len(img.shape) == 4
    img_in = preprocess(img)
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_in.cuda()).float()
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    img_feats = img_feats.half().cpu()
    return img_feats

with torch.no_grad():
    for json_fn in json_fns:
        with open(json_fn, 'r') as fp:
            data_json = json.load(fp)['data']
        for item in tqdm.tqdm(data_json):
            img_fn = item['image']
            img_semiRoot = "/".join(img_fn.split("/")[:-1])
            os.makedirs(os.path.join(save_root, img_semiRoot), exist_ok=True)
            img = torch.from_numpy(np.array(Image.open(os.path.join(img_root, img_fn)).convert('RGB'), dtype=np.float32))/255.0
            feat = clip_embed_img(img.permute(2,0,1).unsqueeze(0)).squeeze(0)
            torch.save(feat, os.path.join(save_root, img_fn+".pt"))
