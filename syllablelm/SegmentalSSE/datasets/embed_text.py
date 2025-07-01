
import tqdm

import json
import os
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


json_fns = ["/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments.json", "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy_with_alignments.json"]
# json_fns = ["/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments.json"]


save_root = "/data/scratch/pyp/datasets/coco_pyp/preextract_embedding_mpnet_text"

os.makedirs(save_root, exist_ok=True)


model = SentenceTransformer('all-mpnet-base-v2').cuda().eval()


with torch.no_grad():
    for json_fn in json_fns:
        with open(json_fn, 'r') as fp:
            data_json = json.load(fp)['data']
        for item in tqdm.tqdm(data_json):
            wav_fn = item['caption']['wav']
            wav_semiRoot = "/".join(wav_fn.split("/")[:-1])
            os.makedirs(os.path.join(save_root, wav_semiRoot), exist_ok=True)
            text = item['caption']['text'].lower()
            feat = torch.from_numpy(model.encode(text)).half()
            torch.save(feat, os.path.join(save_root, wav_fn+".pt"))