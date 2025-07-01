import json
import random
import numpy as np
import os
import torch
import torch.nn.functional
import random
import soundfile as sf
from torch.utils.data import Dataset
import pickle
import torchvision.transforms as transforms
from PIL import Image
import logging
logger = logging.getLogger(__name__)

class SegEmbDataset(Dataset):

    def __init__(self, args, split = "train"):
        self.args = args
        self.split = split
        self.audio_feat_len = args.audio_feat_len if "train" in split else args.val_audio_feat_len
        if split == "train":
            self.audio_dataset_json_file = args.train_audio_dataset_json_file
            boundary_pkl_file = args.train_boundary_pkl_file 
        elif split == "val":
            self.audio_dataset_json_file = args.val_audio_dataset_json_file
            boundary_pkl_file = args.val_boundary_pkl_file
        
        if "librispeech" in args.train_boundary_pkl_file.lower():
            self.root_of_all = "/data1/scratch/datasets_pyp/LibriSpeech/"
            self.audio_base_path = os.path.join(self.root_of_all, "data")
            self.data = []
            print(f"this is training librispeech, and the audio feature length is {self.audio_feat_len}, are you sure???")
            print(f"this is training librispeech, and the audio feature length is {self.audio_feat_len}, are you sure???")
            print(f"this is training librispeech, and the audio feature length is {self.audio_feat_len}, are you sure???")
            if split == "train":
                for folder in ['train-clean-100.json', 'train-clean-360.json', 'train-other-500.json']:
                    with open(os.path.join(self.root_of_all, "vghubert_syllabic_boundary", folder), "r") as f:
                        self.data = self.data + json.load(f)
            elif split == "val":
                folder = "dev-clean.json"
                with open(os.path.join(self.root_of_all, "vghubert_syllabic_boundary", folder), "r") as f:
                    self.data = json.load(f)

        else:
            # self.audio_base_path = "/scratch/cluster/harwath/COCO/SpokenCOCO" if "/scratch/cluster" in self.audio_dataset_json_file else os.path.dirname(self.audio_dataset_json_file)
            if "pseudo_ls" in args.train_boundary_pkl_file.lower():
                self.audio_base_path = ""
            elif "pseudo_codo" in args.train_boundary_pkl_file.lower():
                self.audio_base_path = "/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/"

            with open(self.audio_dataset_json_file, 'r') as fp:
                data_json = json.load(fp)
            self.data = []
            # self.data = self.data[:600]
            
            with open(boundary_pkl_file, "rb") as f:
                boundary = pickle.load(f)
            # print("length of json:", len(self.data))
            # print("length of boundaries: ", len(boundary))
            
            # for i, wav_path in enumerate(boundary):
            #     # if i >= len(self.data):
            #     #     break
            #     json_wav_path = self.data[i]['caption']['wav']
            #     assert wav_path == json_wav_path, f"wav_path in boundary file: {wav_path}, in json file: {json_wav_path}"
            #     self.data[i]['vghubert_boundary'] = boundary[wav_path]['word_boundaries']
                
            for i, datum in enumerate(data_json['data']):
                json_wav_path = datum['caption']['wav']
                assert json_wav_path in boundary, json_wav_path
                if boundary[json_wav_path]['word_boundaries'] is not None:
                    datum['vghubert_boundary'] = boundary[json_wav_path]['word_boundaries']
                    self.data.append(datum)

    def get_fa_b(self, raw_ali):
        if raw_ali == None:
            return None
        b = []
        meta_toks = raw_ali.split()
        for meta_tok in meta_toks:
            toks = meta_tok.split('__')
            if len(toks) == 3:
                b.append([float(toks[0]), float(toks[2])])
        
        if len(b) == 0:
            return None
        return b


    def __len__(self):
        return len(self.data)

    def _LoadAudio(self, path, label_key):
        x, sr = sf.read(path, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        if length_orig > 16000 * self.audio_feat_len:
            audio_length = int(16000 * self.audio_feat_len)
            x = x[:audio_length] 
            if self.args.normalize:
                x_norm = (x - np.mean(x)) / np.std(x)
            else:
                x_norm = x
            x = torch.FloatTensor(x_norm) 
        else:
            audio_length = length_orig
            new_x = torch.zeros(int(16000 * self.audio_feat_len))
            if self.args.normalize:
                x_norm = (x - np.mean(x)) / np.std(x)
                new_x[:audio_length] = torch.FloatTensor(x_norm) 
            else:
                new_x[:audio_length] = torch.FloatTensor(x) 
            x = new_x
        return x, audio_length

    def __getitem__(self, index):

        datum = self.data[index]
        if 'librispeech' in self.args.train_boundary_pkl_file.lower():
            wavpath = os.path.join(self.audio_base_path, datum['wav'])
            label_key = wavpath
        else:
            wavpath = os.path.join(self.audio_base_path, datum['caption']['wav'])
            label_key = datum['caption']['wav'].split(".")[0]
        audio, nframes= self._LoadAudio(wavpath, label_key)
        vghubert_b_in_sec = datum['vghubert_boundary'] if 'vghubert_boundary' in datum else datum['word_boundaries']
        vghubert_b_in_frame = [[round(b[0]/self.args.spf), round(b[1]/self.args.spf)] for b in vghubert_b_in_sec]
        forcealigned_b_in_sec = self.get_fa_b(None if 'text_alignment' not in datum else datum['text_alignment'])
        forcealigned_b_in_frame = None if forcealigned_b_in_sec == None else [[round(b[0]/self.args.spf), round(b[1]/self.args.spf)] for b in forcealigned_b_in_sec]
        #img_id = datum['image'].split("/")[-1].split(".")[0] if "image" in datum else None
        img_id = None
        
        # if self.args.phase not in ['pretrain', 'validate_seg']:
        #     if self.args.anchor_type == "image":
        #         if "/scratch/cluster" in self.audio_dataset_json_file:
        #             anchor_embedding = torch.load(os.path.join("/".join(self.args.exp_dir.split("/")[:-1]), "preextract_embedding_clip_img", datum['image']+".pt"))
        #         else:
        #             anchor_embedding = torch.load(os.path.join("/".join(self.audio_dataset_json_file.split("/")[:-2]), "preextract_embedding_clip_img", datum['image']+".pt"))
        #     elif self.args.anchor_type == "text":
        #         raise NotImplementedError
        #         anchor_embedding = torch.load(os.path.join("/".join(self.audio_dataset_json_file.split("/")[:-2]), "preextract_embedding_mpnet_text", datum['caption']['wav']+".pt"))
        # else:
        anchor_embedding = None


        return audio, nframes, wavpath, vghubert_b_in_sec, forcealigned_b_in_sec, vghubert_b_in_frame, forcealigned_b_in_frame, img_id, anchor_embedding



    def collate(self, batch):
        # audio, padding_mask,
        # tgt, force_aligned_b, these two are always in frame index (i.e. round(sec*50))
        # 'flattened_binary_tgt'
        vals = list(zip(*batch))

        collated = {}
        collated['audio'] = torch.stack(vals[0], dim=0)
        collated['audio_length'] = torch.LongTensor(vals[1])
        collated['wavpath'] = np.array(vals[2])
        collated['padding_mask'] = (torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)).bool()
        # # use sec for evaluation
        if self.args.boundary_tolerance < 1:
            collated['tgt'] = vals[3]
            collated['force_aligned_b'] = vals[4]
        # # use frame for evaluation
        else:
            collated['tgt'] = vals[5]
            collated['force_aligned_b'] = vals[6]
        bts = []
        inners = []
        length = round(16000 * self.audio_feat_len/self.args.downsample_factor)
        tgt_used = vals[5] if not self.args.use_fa_tgt else vals[6]
        for i, b in enumerate(tgt_used):
            bt = torch.zeros((length,))
            if b == None: # only happen when GT word alignment is None
                b = np.unique(vals[3][i])
            else:
                b = np.unique(b)
            b = [t for t in b if t < length]
            bt[b] = 1
            bts.append(bt)
            
            inner = torch.zeros((length,), dtype=int)
            from_pct = 0.33
            to_pct = 0.67
            
            b = sorted(b)
            b.append(length)
            for idx in range(len(b)-1):
                cur_len = b[idx+1] - b[idx]
                if cur_len < 3:
                    inner[b[idx]:b[idx+1]] = 1
                else:
                    inner[b[idx] + int(cur_len * from_pct):b[idx+1] - int(cur_len * (1-to_pct))] = 1
            inners.append(inner)
            
        collated['binary_tgts'] = torch.stack(bts, dim=0)
        collated['binary_tgts_inner_mask'] = torch.stack(inners)
        #collated['img_id'] = np.array(vals[7])
        #if self.args.phase not in ['pretrain', 'validate_seg']:
        #    collated['anchor_embedding'] = torch.stack(vals[8], dim=0)
        collated['anchor_embedding'] = None
        collated['img_id'] = None


        return collated
    