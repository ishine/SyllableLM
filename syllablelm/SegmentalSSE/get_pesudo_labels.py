# load args
# load model
# go through the dataset
# store results
import json
import os
import torch
import numpy as np
import soundfile as sf
import pickle
import tqdm

from models import segmenter


json_fns = ["/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments.json", "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy_with_alignments.json"]
save_pkl_fns = ["/saltpool0/scratch/pyp/discovery/word_unit_discovery/wavlm_crf_1iter/val_data_dict.pkl", "/saltpool0/scratch/pyp/discovery/word_unit_discovery/wavlm_crf_1iter/test_data_dict.pkl", "/saltpool0/scratch/pyp/discovery/word_unit_discovery/wavlm_crf_1iter/train_data_dict.pkl"]
# json_fns = ["/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json"]
# save_pkl_fns = ["/saltpool0/scratch/pyp/discovery/word_unit_discovery/wavlm_crf_1iter/val_data_dict.pkl"]
# now hard code args
# hard code this as well
os.makedirs("/saltpool0/scratch/pyp/discovery/word_unit_discovery/wavlm_crf_1iter", exist_ok=True)

exp_fn = "/data/scratch/pyp/exp_pyp/SegmentalSSE/a40_crf1_crfWei1_bceWei0_freezeSeg0_pretrain_faTrain0_posWeight3_useFa0_segQuan0.95_ancimage_cls1_comLayer1_9_clsPolicy0.00001_featPolicy0_clsMatch4_featMatch0"
wav_root = "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO"
batch_size = 100
spf = 0.02
seg_quantile_threshold=0.95
max_length = int(15*16000)


# load args and model
with open(os.path.join(exp_fn, "args.pkl"), "rb") as arg_fn:
    model_args = pickle.load(arg_fn)
model = segmenter.Segmenter(model_args)
model.load_state_dict(torch.load(os.path.join(exp_fn, "best_bundle.pth"))['model_seg'])
model.eval()
model.cuda()

## get max mem
with torch.no_grad():
    audio = torch.randn((batch_size, max_length))
    padding_mask = torch.zeros((batch_size, max_length))
    _ = model(audio, padding_mask, decode=True)

    # json_fn = ""
    for json_fn, save_pkl_fn in zip(json_fns, save_pkl_fns):
        with open(json_fn, "r") as f:
            data = json.load(f)['data']
        all_b = {}
        batch = {"audio": [], "lengths": [], "padding_mask": None, "wav":[]}
        for jj, item in enumerate(tqdm.tqdm(data)):
            wav_path = item['caption']['wav']
            wav_fn = os.path.join(wav_root, wav_path)
            x, sr = sf.read(wav_fn, dtype = 'float32')
            assert sr == 16000
            x = x[:max_length]
            if len(batch['audio']) < batch_size:
                batch['audio'].append(torch.from_numpy(x))
                batch['lengths'].append(len(x))
                batch['wav'].append(wav_path)
            else:
                batch['audio'] = torch.nn.utils.rnn.pad_sequence(batch['audio'], batch_first=True)
                batch['padding_mask'] = torch.arange(len(batch['audio'][0])).unsqueeze(0) >= torch.LongTensor(batch['lengths']).unsqueeze(1)
                # print("shape of audio: ", batch['audio'].shape)
                # print("shape of padding_mask: ", batch['padding_mask'].shape)
                out = model(batch['audio'], batch['padding_mask'], decode=True)
                for out_wav_fn, crf_pred, length, logit in zip(batch['wav'], out['crf_infer'], out['lengths'], out['logits']):
                    pred_boundaries = np.where(crf_pred[:length].cpu().numpy())[0].tolist()
                    pred_boundaries = [pb*spf for pb in pred_boundaries]
                    if len(pred_boundaries) <= 1:
                        # use thresholding binary prediction
                        logit = logit[:length]
                        # prob = torch.sigmoid(logit).cpu().float()
                        # [T, 2], the second one is prob for boundary
                        prob = torch.softmax(logit, dim=1)[:, 1].cpu().float()
                        thres = torch.quantile(prob, seg_quantile_threshold)
                        pred_binary_b = (prob >= thres).int().numpy()
                        pred_boundaries = np.where(pred_binary_b)[0]
                        pred_boundaries = [pb*spf for pb in pred_boundaries]
                        if len(pred_boundaries) == 0:
                            pred_boundaries = [pb*spf for pb in list(range(length))[::20]]
                    all_b[out_wav_fn] = {'word_boundaries':[[l,r] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]}
                batch = {"audio":[torch.from_numpy(x)], "lengths": [len(x)], "padding_mask": None, "wav": [wav_path]}
        if len(batch['audio']) > 0:
            print(f"operate on the last {len(batch['audio'])} files")
            batch['audio'] = torch.nn.utils.rnn.pad_sequence(batch['audio'], batch_first=True)
            batch['padding_mask'] = torch.arange(len(batch['audio'][0])).unsqueeze(0) >= torch.LongTensor(batch['lengths']).unsqueeze(1)
            out = model(batch['audio'], batch['padding_mask'], decode=True)
            for out_wav_fn, crf_pred, length, logit in zip(batch['wav'], out['crf_infer'], out['lengths'], out['logits']):
                pred_boundaries = np.where(crf_pred[:length].cpu().numpy())[0].tolist()
                pred_boundaries = [pb*spf for pb in pred_boundaries]
                if len(pred_boundaries) <= 1:
                    # use thresholding binary prediction
                    logit = logit[:length]
                    # prob = torch.sigmoid(logit).cpu().float()
                    # [T, 2], the second one is prob for boundary
                    prob = torch.softmax(logit, dim=1)[:, 1].cpu().float()
                    thres = torch.quantile(prob, seg_quantile_threshold)
                    pred_binary_b = (prob >= thres).int().numpy()
                    pred_boundaries = np.where(pred_binary_b)[0]
                    pred_boundaries = [pb*spf for pb in pred_boundaries]
                    if len(pred_boundaries) == 0:
                        pred_boundaries = [pb*spf for pb in list(range(length))[::20]]
                all_b[out_wav_fn] = {'word_boundaries':[[l,r] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]}
        # for ii, key in enumerate(all_b):
        #     print(all_b[key])
        #     if ii > 30:
        #         break
        print(f"save the detected boundaries at {save_pkl_fn}")
        with open(save_pkl_fn, "wb") as sp:
            pickle.dump(all_b, sp)