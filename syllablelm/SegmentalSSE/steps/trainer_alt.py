import time
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import spokencoco_dataset, places_dataset
from datasets.sampler import StatefulSampler
from models import embedder, segmenter
from .utils import *
from .trainer_utils import *
from .bert_adam import BertAdam
from apex.fp16_utils import *
from apex import amp
from logging import getLogger
logger = getLogger(__name__)

class Trainer:
    # this trainer will alternate the optimization of segmenter and combiner 
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()
        self.progress, self.total_progress = setup_progress(self)
        self.model_seg, self.model_emb, self.trainables, self.indices, self.optim_states = self._setup_models()
        self.train_loader, self.valid_loader, self.valid_loader2, self.train_sampler, self.train_data_length = self._setup_dataloader()
        self.total_num_updates = int(math.floor(self.train_data_length / self.args.batch_size))*self.args.n_epochs
        self.optimizer = self._setup_optimizer()
        [self.model_seg, self.model_emb], [a,b] = amp.initialize(models=[self.model_seg, self.model_emb], optimizers=[self.optimizer['seg'], self.optimizer['emb']], opt_level=self.args.opt_level)
        if torch.cuda.device_count() > 1:
            self.model_seg = nn.DataParallel(self.model_seg)
            self.model_emb = nn.DataParallel(self.model_emb)
        self.scheduler = self._setup_scheduler()
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([self.args.positive_weight]).to(self.device)) if self.args.phase == "pretrain" else Margin_InfoNCE_loss
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, self.args.positive_weight]).to(self.device)) if self.args.phase == "pretrain" else Margin_InfoNCE_loss
        logger.info(f"batch size: {self.args.batch_size}")

    def train(self):
        flag = True
        step_per_epoch = int(self.train_data_length/self.args.batch_size)
        data_start_time = time.time()

        while flag:
            for i, batch in enumerate(self.train_loader):
                data_end_time = time.time()
                self.model_seg.train()
                self.model_emb.train()
                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    self.validate_and_save()
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.optimizer['emb'].get_lr())

                self.writer.add_scalar("lr_emb", cur_lr, self.progress['num_updates'])
                self.writer.add_scalar("lr_seg", np.mean(self.optimizer['seg'].get_lr()), self.progress['num_updates'])
                cur_step = self.progress['num_updates'] % step_per_epoch

                losses = self.forward(batch)

                for key in losses:
                    if key in self.meters:
                        self.meters[key].update(losses[key].mean().cpu().item(), batch['audio'].shape[0])
                        self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                weighted_loss = self.weight_loss(losses)

                self.meters['weighted_loss'].update(weighted_loss.item(), batch['audio'].shape[0])
                self.writer.add_scalar('weighted_loss', weighted_loss.item(), self.progress['num_updates'])
                if self.progress['num_updates'] % 2 and  self.progress['num_updates'] >= self.args.freeze_segmenter:
                    with amp.scale_loss(weighted_loss, self.optimizer['seg']) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer['seg']), 1.)
                    self.optimizer['seg'].step()
                else:
                    with amp.scale_loss(weighted_loss, self.optimizer['emb']) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer['emb']), 1.)
                    # manually zero out the gradient in segmenter
                    if self.progress['num_updates'] < self.args.freeze_segmenter:
                        for p in self.model_seg.parameters():
                            if p.requires_grad:
                                p.grad.data.zero_()
                    self.optimizer['emb'].step()

                self.optimizer['seg'].zero_grad()
                self.optimizer['emb'].zero_grad()
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])

                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.7f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                # validation and save models
                if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    self.validate_and_save(places=self.args.places)

                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()

    def forward(self, batch):
        if self.args.phase == "pretrain":
            losses = {}
            out = self.model_seg(batch['audio'], padding_mask = batch['padding_mask'], tgt = batch['binary_tgts']) # flattened_logitsshould be flattened, with padded tokens taken out
            flattened_logits = []
            flattened_binary_tgts = []
            for logit, length, binary_tgt in zip(out['logits'], out['lengths'], batch['binary_tgts']):
                flattened_logits.append(logit[:length])
                flattened_binary_tgts.append(binary_tgt[:length])
            flattened_logits = torch.cat(flattened_logits, dim=0).squeeze(1)
            # flattened_binary_tgts = torch.cat(flattened_binary_tgts, dim=0)
            flattened_binary_tgts = torch.cat(flattened_binary_tgts, dim=0).long()
            
            # assert len(flattened_binary_tgts.shape) == 1 and len(flattened_logits.shape) == 1 and len(flattened_binary_tgts) == len(flattened_logits), f"flattened_binary_tgts.shape: {flattened_binary_tgts.shape}, flattened_logits.shape: {flattened_logits.shape}"
            assert len(flattened_binary_tgts.shape) == 1 and len(flattened_logits.shape) == 2 and len(flattened_binary_tgts) == len(flattened_logits), f"flattened_binary_tgts.shape: {flattened_binary_tgts.shape}, flattened_logits.shape: {flattened_logits.shape}"

            losses['bce_loss'] = self.criterion(flattened_logits, flattened_binary_tgts.to(flattened_logits.device))

            # CRF
            if self.args.crf:
                losses['crf_loss'] = out['crf_loss'].mean()
                
        else:
            # raise NotImplementedError("haven't implement how to deal with [B,T,2] output shape")
            assert self.args.phase == "train", self.args.phase
            assert self.args.seg_cap != 0 # cap a seg if it's too long
            # print(type(self.model_seg))
            if type(self.model_seg) == nn.Identity or (type(self.model_seg) == nn.DataParallel and type(self.model_seg.module) == torch.nn.Identity):
                all_seg = []
                log_probs = None
                all_len = ((~batch['padding_mask']).sum(-1)/self.args.downsample_factor).int()
                for i in range(len(batch['force_aligned_b'])):
                    if batch['force_aligned_b'][i] != None:
                        seg = [[l, min(l+self.args.seg_cap, r)] for l, r in batch['force_aligned_b'][i]]
                    else:
                        cur_len = all_len[i]
                        seg = [[int(l), int(min(l+self.args.seg_cap, r))] for l, r in zip(list(range(cur_len))[::25][:-1], list(range(cur_len))[::25][1:])]
                    all_seg.append(seg)
                out = self.model_emb(batch['audio'], all_seg) # TODO not sure if copying conv_feats will mess up the gradient
            else:
                assert self.args.seg_quantile_threshold != 0
                # if self.progress['num_updates'] < self.args.freeze_segmenter:
                #     with torch.no_grad():
                #         seg_out = self.model_seg(batch['audio'], padding_mask = batch['padding_mask'],tgt = batch['binary_tgts'], decode=True, rescore=True) # flattened_logitsshould be flattened, with padded tokens taken out
                # else:
                seg_out = self.model_seg(batch['audio'], padding_mask = batch['padding_mask'],tgt = batch['binary_tgts'], decode=True, rescore=True) # flattened_logitsshould be flattened, with padded tokens taken out
                all_seg = []
                log_probs = []
                for i, (crf_pred, llh, length) in enumerate(zip(seg_out['crf_infer'], seg_out['crf_llh'], seg_out['lengths'])):
                    log_probs.append(llh)
                    pred_boundaries = np.where(crf_pred[:length].detach().cpu().numpy())[0]
                    seg = [[l.item(),r.item()] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]
                    seg = [[l, min(l+self.args.seg_cap, r)] for l,r in seg] # always take the left part of the seg, as the start of a word is more important than the end of a word
                    if len(seg) == 0:
                        print("During training, crf predicted boundaries is of zero length!")
                        # seg = [[int(l), int(min(l+self.args.seg_cap, r))] for l, r in zip(list(range(length))[::25][:-1], list(range(length))[::25][1:])]
                        seg = [[0, 25]]
                    all_seg.append(seg)
                out = self.model_emb(batch['audio'] if 'conv_feats' not in seg_out else seg_out['conv_feats'], all_seg) # TODO not sure if copying conv_feats will mess up the gradient
            # print(log_probs)


            
            # print(f"from trainer line 144, cls token shape: ", out['cls_token'].shape)
            losses = {}
            losses['crf_loss'] = seg_out['crf_loss'].mean()
            if self.args.cls_matching_weight != 0 or self.args.cls_policy_gradient_weight != 0:
                coarse_cross_relationship_score_matrix = batch['anchor_embedding'].to(out['cls_token']) @ out['cls_token'].transpose(0,1)
                cls_matching_loss = self.criterion(coarse_cross_relationship_score_matrix, margin=0, img_id = batch['img_id']) # TODO now fix margin at 0
                losses['cls_matching_loss'] = cls_matching_loss.mean()
                if log_probs != None:
                    assert len(cls_matching_loss) == len(log_probs), f"cls_matching_loss length: {len(cls_matching_loss)}, log_probs length: {len(log_probs)}"
                    if self.args.running_average != 0:
                        losses['cls_policy_gradient_loss'] = sum([(cls_matching_loss[i].detach() - self.meters['cls_matching_loss'].avg) * log_probs[i] for i in range(len(log_probs))])/len(log_probs) # reward shouldn't be differentiable
                    else:
                        losses['cls_policy_gradient_loss'] = sum([cls_matching_loss[i].detach() * log_probs[i] for i in range(len(log_probs))])/len(log_probs) # reward shouldn't be differentiable
                # print([cls_matching_loss[i].detach() for i in range(len(log_probs))])
            if self.args.feat_matching_weight != 0 or self.args.feat_policy_gradient_weight != 0:
                coarse_cross_relationship_score_matrix = batch['anchor_embedding'].to(out['meanpool_feature']) @ out['meanpool_feature'].transpose(0,1)
                feat_matching_loss = self.criterion(coarse_cross_relationship_score_matrix, margin=0, img_id = batch['img_id'])
                losses['feat_matching_loss'] = feat_matching_loss.mean()
                if log_probs != None:
                    assert len(feat_matching_loss) == len(log_probs), f"feat_matching_loss length: {len(feat_matching_loss)}, log_probs length: {len(log_probs)}"
                    if self.args.running_average != 0:
                        losses['feat_policy_gradient_loss'] = sum([(feat_matching_loss[i].detach() - self.meters['feat_matching_loss'].avg) * log_probs[i] for i in range(len(log_probs))])/len(log_probs) # reward shouldn't be differentiable
                    else:
                        losses['feat_policy_gradient_loss'] = sum([feat_matching_loss[i].detach() * log_probs[i] for i in range(len(log_probs))])/len(log_probs) # reward shouldn't be differentiable
            # print(losses)
            #     seg_out = self.model_seg(batch['audio'], padding_mask = batch['padding_mask']) # flattened_logitsshould be flattened, with padded tokens taken out
            #     all_seg = []
            #     log_probs = []
            #     for i, (logit, length) in enumerate(zip(seg_out['logits'], seg_out['lengths'])):
            #         prob = torch.sigmoid(logit[:length])
            #         # log_probs.append(log_prob.sum())
            #         thres = torch.quantile(prob.float(), self.args.seg_quantile_threshold)
            #         pred_boundaries, pred_not_boundaries = torch.where(prob >= thres)[0], torch.where(prob < thres)[0]

            #         b_log_prob, nb_log_prob = torch.log(prob[pred_boundaries]), torch.log(1-prob[pred_not_boundaries]+1e-7)
            #         log_probs.append(b_log_prob.sum() + nb_log_prob.sum())

            #         seg = [[l.detach().item(),r.detach().item()] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]
            #         seg = [[l, min(l+self.args.seg_cap, r)] for l,r in seg] # always take the left part of the seg, as the start of a word is more important than the end of a word
            #         all_seg.append(seg)
            #     out = self.model_emb(batch['audio'] if 'conv_feats' not in seg_out else seg_out['conv_feats'], all_seg) # TODO not sure if copying conv_feats will mess up the gradient
            # # print(log_probs)


            
            # # print(f"from trainer line 144, cls token shape: ", out['cls_token'].shape)
            # losses = {}
            # if self.args.cls_matching_weight != 0 or self.args.cls_policy_gradient_weight != 0:
            #     coarse_cross_relationship_score_matrix = batch['anchor_embedding'].to(out['cls_token']) @ out['cls_token'].transpose(0,1)
            #     cls_matching_loss = self.criterion(coarse_cross_relationship_score_matrix, margin=0, img_id = batch['img_id']) # TODO now fix margin at 0
            #     losses['cls_matching_loss'] = cls_matching_loss.mean()
            #     if log_probs != None:
            #         assert len(cls_matching_loss) == len(log_probs), f"cls_matching_loss length: {len(cls_matching_loss)}, log_probs length: {len(log_probs)}"
            #         losses['cls_policy_gradient_loss'] = sum([cls_matching_loss[i].detach() * log_probs[i] for i in range(len(log_probs))])/len(log_probs) # reward shouldn't have parameters
            #     # print([cls_matching_loss[i].detach() for i in range(len(log_probs))])
            # if self.args.feat_matching_weight != 0 or self.args.feat_policy_gradient_weight != 0:
            #     coarse_cross_relationship_score_matrix = batch['anchor_embedding'].to(out['meanpool_feature']) @ out['meanpool_feature'].transpose(0,1)
            #     feat_matching_loss = self.criterion(coarse_cross_relationship_score_matrix, margin=0, img_id = batch['img_id'])
            #     losses['feat_matching_loss'] = feat_matching_loss.mean()
            #     if log_probs != None:
            #         assert len(feat_matching_loss) == len(log_probs), f"feat_matching_loss length: {len(feat_matching_loss)}, log_probs length: {len(log_probs)}"
            #         losses['feat_policy_gradient_loss'] = sum([feat_matching_loss[i].detach() * log_probs[i] for i in range(len(log_probs))])/len(log_probs) # reward shouldn't have parameters
            # # print(losses)
        return losses

    def validate_and_save(self, libri=False, places=False):
        self.model_seg.eval()
        self.model_emb.eval()

        if self.args.phase == "pretrain":
            seg_f1, seg_prec, seg_rec = self.validate_seg(self.valid_loader)
            r1 = 0
        else:
            r10, r5, r1 = self.validate(self.valid_loader)
            if type(self.model_seg) == nn.Identity or (type(self.model_seg) == nn.DataParallel and type(self.model_seg.module) == torch.nn.Identity):
                seg_f1, seg_prec, seg_rec = 0,0,0
            else:
                seg_f1, seg_prec, seg_rec = self.validate_seg(self.valid_loader)
        
        # r1 = 0.1 # ignore validation, for debugging
        # TODO this might need more work
        if r1 > self.progress['best_acc'] or seg_f1 > self.progress['best_f1']:
            self.progress['best_epoch'] = self.progress['epoch']
            self.progress['best_acc'] = r1
            self.progress['best_f1'] = seg_f1
            self.progress['prec_of_bestf1'] = seg_prec
            self.progress['rec_of_bestf1'] = seg_rec
            save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
            torch.save(
                {
                    "model_seg": self.model_seg.module.state_dict() if torch.cuda.device_count() > 1 else self.model_seg.state_dict(),
                    "model_emb": self.model_emb.module.state_dict() if torch.cuda.device_count() > 1 else self.model_emb.state_dict(),
                    "optimizer":  {"seg": self.optimizer['seg'].state_dict(), "emb": self.optimizer['emb'].state_dict()},
                    "indices": self.train_sampler.state_dict()
                },save_path
            )
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
        save_progress(self)
        save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        torch.save(
            {
                "model_seg": self.model_seg.module.state_dict() if torch.cuda.device_count() > 1 else self.model_seg.state_dict(),
                "model_emb": self.model_emb.module.state_dict() if torch.cuda.device_count() > 1 else self.model_emb.state_dict(),
                "optimizer":  {"seg": self.optimizer['seg'].state_dict(), "emb": self.optimizer['emb'].state_dict()},
                "indices": self.train_sampler.state_dict()
            },save_path
        )
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")
    
    def validate_seg(self, valid_loader=None, hide_progress=True):
        if valid_loader == None:
            valid_loader = self.valid_loader
        self.model_seg.eval()
        self.model_emb.eval()

        start_val_time = time.time()
        with torch.no_grad():
            match_pseudo_pred_count = 0
            match_pred_pseudo_count = 0
            pseudo_b_len = 0
            pred_b_len = 0
            match_gt_pred_count = 0
            match_pred_gt_count = 0
            gt_b_len = 0
            pred_gt_b_len = 0
            match_gt_crfPred_count = 0
            match_crfPred_gt_count = 0
            gt_crfPred_b_len = 0
            crfPred_gt_b_len = 0
            match_pseudo_crfPred_count = 0
            match_crfPred_pseudo_count = 0
            pseudo_crfPred_b_len = 0
            crfPred_pseudo_b_len = 0
            for batch in tqdm(valid_loader, disable=hide_progress):
                out = self.model_seg(batch['audio'], batch['padding_mask'], decode=True)
                # out = {} # test pseudo label acc
                # out['lengths'] = batch['tgt']
                # out['logits'] = batch['tgt']
                # out['crf_infer'] = batch['tgt']
                for gt_boundaries, tgt, logit, length in zip(batch['force_aligned_b'], batch['tgt'], out['logits'], out['lengths']):
                    logit = logit[:length]
                    # prob = torch.sigmoid(logit).cpu().float()
                    # [T, 2], the second one is prob for boundary
                    prob = torch.softmax(logit, dim=1)[:, 1].cpu().float()
                    if self.args.seg_quantile_threshold != 0:
                        thres = torch.quantile(prob, self.args.seg_quantile_threshold)
                    else:
                        thres = self.args.seg_threshold
                    pred_binary_b = (prob >= thres).int().numpy()
                    pred_boundaries = np.where(pred_binary_b)[0]
                    if self.args.boundary_tolerance < 1:
                        # use sec, rather than frame for seg evaluation
                        pred_boundaries = [pb*self.args.spf for pb in pred_boundaries]
                    if len(pred_boundaries) == 0:
                        pred_boundaries = [0]

                    pseudo_boundaries = np.unique(tgt)
                    # pred_boundaries = pseudo_boundaries # test pseudo label acc
                    
                    a, b, c, d = find_boundary_matches(pseudo_boundaries, pred_boundaries, self.args.boundary_tolerance) # exclude the first and last boundary from GT boundaries, these are already excluded in pred_boundaries
                    match_pseudo_pred_count += a
                    match_pred_pseudo_count += b
                    pseudo_b_len += c
                    pred_b_len += d
            

                    if gt_boundaries != None:
                        gt_boundaries = np.unique(gt_boundaries)
                        a, b, c, d = find_boundary_matches(gt_boundaries[1:-1] if len(gt_boundaries) > 2 else gt_boundaries, pred_boundaries[1:-1] if len(pred_boundaries) > 2 else pred_boundaries, self.args.boundary_tolerance) # exclude the first and last boundary from GT boundaries, these are already excluded in pred_boundaries
                        match_gt_pred_count += a
                        match_pred_gt_count += b
                        gt_b_len += c
                        pred_gt_b_len += d
                if self.args.crf and self.args.crf_infer:
                    for tgt, gt_boundaries, tgt, crf_pred, length in zip(batch['tgt'], batch['force_aligned_b'], batch['tgt'], out['crf_infer'], out['lengths']):
                        pred_boundaries = np.where(crf_pred[:length].cpu().numpy())[0]
                        if self.args.boundary_tolerance < 1:
                            # use sec, rather than frame for seg evaluation
                            pred_boundaries = [pb*self.args.spf for pb in pred_boundaries]
                        pseudo_boundaries = np.unique(tgt)
                        # pred_boundaries = pseudo_boundaries # test pseudo label acc
                        a, b, c, d = find_boundary_matches(pseudo_boundaries, pred_boundaries, self.args.boundary_tolerance) # exclude the first and last boundary from GT boundaries, these are already excluded in pred_boundaries
                        match_pseudo_crfPred_count += a
                        match_crfPred_pseudo_count += b
                        pseudo_crfPred_b_len += c
                        crfPred_pseudo_b_len += d
                        if gt_boundaries != None:
                            gt_boundaries = np.unique(gt_boundaries)
                            a, b, c, d = find_boundary_matches(gt_boundaries[1:-1] if len(gt_boundaries) > 2 else gt_boundaries, pred_boundaries[1:-1] if len(pred_boundaries) > 2 else pred_boundaries, self.args.boundary_tolerance) # exclude the first and last boundary from GT boundaries, these are already excluded in pred_boundaries
                            match_gt_crfPred_count += a
                            match_crfPred_gt_count += b
                            gt_crfPred_b_len += c
                            crfPred_gt_b_len += d


            logger.info(f"validation time: {time.time() - start_val_time:.3f}")
            b_prec = match_pred_pseudo_count / (pred_b_len +1e-7)
            b_recall = match_pseudo_pred_count / (pseudo_b_len  +1e-7)
            b_f1 = 2* b_prec * b_recall / (b_prec + b_recall +1e-7)
            b_os = b_recall / (b_prec +1e-7) - 1.
            b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
            b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
            b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
            logger.info("With VGHuBERT as GT")
            logger.info(f"F1: [{b_f1:.5f}], Prec: [{b_prec:.5f}], Recall: [{b_recall:.5f}]")
            logger.info(f"Over-segmentation: [{b_os: .5f}], R-value: [{b_r_val: .5f}]")

            b_prec2 = match_pred_gt_count / (pred_gt_b_len +1e-7)
            b_recall2 = match_gt_pred_count / (gt_b_len +1e-7)
            b_f1_2 = 2* b_prec2 * b_recall2 / (b_prec2 + b_recall2+1e-7)
            b_os_2 = b_recall2 / (b_prec2+1e-7) - 1.
            b_r1_2 = np.sqrt((1-b_recall2)**2 + b_os_2**2)
            b_r2_2 = (-b_os_2 + b_recall2 - 1) / np.sqrt(2)
            b_r_val_2 = 1. - (np.abs(b_r1_2) + np.abs(b_r2_2))/2.
            logger.info("With Force Aligned Boundaries as GT")
            logger.info(f"F1: [{b_f1_2:.5f}], Prec: [{b_prec2:.5f}], Recall: [{b_recall2:.5f}]")
            logger.info(f"Over-segmentation: [{b_os_2: .5f}], R-value: [{b_r_val_2: .5f}]")

            self.writer.add_scalar("F1_VGHuBERT", b_f1, self.progress['num_updates'])
            self.writer.add_scalar("Precision_VGHuBERT", b_prec, self.progress['num_updates'])
            self.writer.add_scalar("Recall_VGHuBERT", b_recall, self.progress['num_updates'])
            self.writer.add_scalar("R_value_VGHuBERT", b_r_val, self.progress['num_updates'])
            self.writer.add_scalar("F1_ForceAligned", b_f1_2, self.progress['num_updates'])
            self.writer.add_scalar("Precision_ForceAligned", b_prec2, self.progress['num_updates'])
            self.writer.add_scalar("Recall_ForceAligned", b_recall2, self.progress['num_updates'])
            self.writer.add_scalar("R_value_ForceAligned", b_r_val_2, self.progress['num_updates'])
            
            if self.args.crf and self.args.crf_infer:
                b_prec3 = match_crfPred_pseudo_count / (crfPred_pseudo_b_len  +1e-7)
                b_recall3 = match_pseudo_crfPred_count / (pseudo_crfPred_b_len  +1e-7)
                b_f1_3 = 2* b_prec3 * b_recall3 / (b_prec3 + b_recall3+1e-7)
                b_os_3 = b_recall3 / (b_prec3+1e-7) - 1.
                b_r1_3 = np.sqrt((1-b_recall3)**2 + b_os_3**2)
                b_r2_3 = (-b_os_3 + b_recall3 - 1) / np.sqrt(2)
                b_r_val_3 = 1. - (np.abs(b_r1_3) + np.abs(b_r2_3))/2.
                logger.info("Using CRF Pred, With VG-HuBERT Boundaries as GT")
                logger.info(f"F1: [{b_f1_3:.5f}], Prec: [{b_prec3:.5f}], Recall: [{b_recall3:.5f}]")
                logger.info(f"Over-segmentation: [{b_os_3: .5f}], R-value: [{b_r_val_3: .5f}]")
                self.writer.add_scalar("CRF_F1_VGHuBERT", b_f1_3, self.progress['num_updates'])
                self.writer.add_scalar("CRF_Precision_VGHuBERT", b_prec3, self.progress['num_updates'])
                self.writer.add_scalar("CRF_Recall_VGHuBERT", b_recall3, self.progress['num_updates'])
                self.writer.add_scalar("CRF_R_value_VGHuBERT", b_r_val_3, self.progress['num_updates'])

                b_prec4 = match_crfPred_gt_count / (crfPred_gt_b_len + 1e-7)
                b_recall4 = match_gt_crfPred_count / (gt_crfPred_b_len +1e-7)
                b_f1_4 = 2* b_prec4 * b_recall4 / (b_prec4 + b_recall4+1e-7)
                b_os_4 = b_recall4 / (b_prec4+1e-7) - 1.
                b_r1_4 = np.sqrt((1-b_recall4)**2 + b_os_4**2)
                b_r2_4 = (-b_os_4 + b_recall4 - 1) / np.sqrt(2)
                b_r_val_4 = 1. - (np.abs(b_r1_4) + np.abs(b_r2_4))/2.
                logger.info("Using *CRF Pred*, With Force Aligned Boundaries as GT")
                logger.info(f"F1: [{b_f1_4:.5f}], Prec: [{b_prec4:.5f}], Recall: [{b_recall4:.5f}]")
                logger.info(f"Over-segmentation: [{b_os_4: .5f}], R-value: [{b_r_val_4: .5f}]")
                self.writer.add_scalar("CRF_F1_ForceAligned", b_f1_4, self.progress['num_updates'])
                self.writer.add_scalar("CRF_Precision_ForceAligned", b_prec4, self.progress['num_updates'])
                self.writer.add_scalar("CRF_Recall_ForceAligned", b_recall4, self.progress['num_updates'])
                self.writer.add_scalar("CRF_R_value_ForceAligned", b_r_val_4, self.progress['num_updates'])
                return b_f1_3, b_prec3, b_recall3

            return b_f1, b_prec, b_recall


    def validate(self, valid_loader=None, unseen=False, hide_progress=True):
        if valid_loader == None:
            valid_loader = self.valid_loader
        self.model_seg.eval()
        self.model_emb.eval()

        start_val_time = time.time()
        N_examples = valid_loader.dataset.__len__()

        # frame_counts = []
        with torch.no_grad():
            # get single modal representations
            audio_feats_total = []
            audio_cls_total = []
            audio_img_id_total = [] # this is same order as audio_cls_total and audio_feats_total
            img_id_to_img_feats = {}
            img_img_id_list = []
            img_cls_list = [] # this is distinct, order is the same as img_img_id_list
            img_feats_list = [] # this is distinct, order is the same as img_img_id_list
            for i, batch in enumerate(tqdm(valid_loader, disable=hide_progress)):
                self.model_seg.eval()
                self.model_emb.eval()
                
                assert self.args.seg_quantile_threshold != 0
                if type(self.model_seg) == nn.Identity or (type(self.model_seg) == nn.DataParallel and type(self.model_seg.module) == torch.nn.Identity):
                    all_seg = []
                    log_probs = None
                    all_len = ((~batch['padding_mask']).sum(-1)/self.args.downsample_factor).int()
                    for k in range(len(batch['force_aligned_b'])):
                        if batch['force_aligned_b'][k] != None:
                            seg = [[l, min(l+self.args.seg_cap, r)] for l, r in batch['force_aligned_b'][k]]
                        else:
                            cur_len = all_len[k]
                            seg = [[int(l), int(min(l+self.args.seg_cap, r))] for l, r in zip(list(range(cur_len))[::25][:-1], list(range(cur_len))[::25][1:])]
                        all_seg.append(seg)
                    out = self.model_emb(batch['audio'], all_seg) # TODO not sure if copying conv_feats will mess up the gradient
                else:
                    seg_out = self.model_seg(batch['audio'], padding_mask = batch['padding_mask']) # flattened_logitsshould be flattened, with padded tokens taken out
                    all_seg = []
                    for i, (crf_pred, length) in enumerate(zip(seg_out['crf_infer'], seg_out['lengths'])):
                        pred_boundaries = np.where(crf_pred[:length].detach().cpu().numpy())[0]
                    # seg_out = self.model_seg(batch['audio'], padding_mask = batch['padding_mask']) # flattened_logitsshould be flattened, with padded tokens taken out
                    # all_seg = []
                    # log_probs = []
                    # for k, (logit, length) in enumerate(zip(seg_out['logits'], seg_out['lengths'])):
                    #     log_prob = F.logsigmoid(logit[:length])
                    #     log_probs.append(log_prob.sum())
                    #     assert self.args.seg_quantile_threshold != 0

                        # thres = torch.quantile(log_prob.float(), self.args.seg_quantile_threshold)
                        # pred_boundaries = torch.where(log_prob >= thres)[0]
                        # seg = [[l.detach().item(),r.detach().item()] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]
                        seg = [[l.item(),r.item()] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]
                        seg = [[l, min(l+self.args.seg_cap, r)] for l,r in seg] # always take the left part of the seg, as the start of a word is more important than the end of a word
                        if len(seg) == 0:
                            print("During inference, crf predicted boundaries is of zero length!")
                            # seg = [[int(l), int(min(l+self.args.seg_cap, r))] for l, r in zip(list(range(length))[::25][:-1], list(range(length))[::25][1:])]
                            seg = [[0, 25]]
                        all_seg.append(seg)
                    out = self.model_emb(batch['audio'] if 'conv_feats' not in seg_out else seg_out['conv_feats'], all_seg) # TODO not sure if copying conv_feats will mess up the gradient
                # print(log_probs)

                # for k, (logit, length) in enumerate(zip(seg_out['logits'], seg_out['lengths'])):
                #     log_prob = F.logsigmoid(logit[:length])
                #     thres = torch.quantile(log_prob.float(), self.args.seg_quantile_threshold)
                #     pred_boundaries = torch.where(log_prob >= thres)[0]
                #     # seg = [[l.detach().item(),r.detach().item()] for l, r in zip(pred_boundaries[:-1], pred_boundaries[1:])]
                #     if batch['force_aligned_b'][k] != None:
                #         seg = batch['force_aligned_b'][k]
                #     else:
                #         seg = [[int(l), int(r)] for l, r in zip(list(range(len(log_prob)))[::25][:-1], list(range(len(log_prob)))[::25][1:])]
                #     if self.args.seg_cap != 0: # cap a seg if it's too long
                #         seg = [[l, min(l+self.args.seg_cap, r)] for l,r in seg] # always take the left part of the seg, as the start of a word is more important than the end of a word
                #     all_seg.append(seg)

                # out = self.model_emb(batch['audio'] if 'conv_feats' not in seg_out else seg_out['conv_feats'], all_seg)
                audio_cls_total.append(out['cls_token'])
                audio_feats_total.append(out['meanpool_feature'])

                audio_img_id_total.append(batch['img_id'])
                visual_feats = batch['anchor_embedding'].to(out['cls_token'])
                visual_cls = visual_feats
                for i, img_id in enumerate(batch['img_id']):
                    if img_id not in img_id_to_img_feats:
                        img_id_to_img_feats[img_id] = 1
                        img_feats_list.append(visual_feats[i])
                        img_cls_list.append(visual_cls[i] if visual_cls is not None else None)
                        img_img_id_list.append(img_id)
                
            logger.info(f"time can be cached: {time.time() - start_val_time:.3f}")
            audio_cls_total = torch.cat(audio_cls_total) if audio_cls_total[0] is not None else None
            img_cls_list = torch.stack(img_cls_list) if img_cls_list[0] is not None else None
            audio_feats_total = torch.cat(audio_feats_total)
            img_feats_list = torch.stack(img_feats_list)
            audio_img_id_total = np.concatenate(audio_img_id_total)
            img_img_id_list = np.array(img_img_id_list)
            if self.args.cls_matching_weight > 0:
                coarse_cross_relationship_score_matrix = img_cls_list @ audio_cls_total.transpose(0,1)
                # print(coarse_cross_relationship_score_matrix.shape)
                recalls_cls = calc_recalls_from_S_one_to_many_coarse(coarse_cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)
                avg_acc_coarse = (recalls_cls['A_r10'] + recalls_cls['I_r10']) / 2
                avg_acc_r1_coarse = (recalls_cls['A_r1'] + recalls_cls['I_r1']) / 2
                if unseen:
                    logger.info("UNSEEN UNSEEN UNSEEN")
                    self.writer.add_scalar("acc_coarse_cls_unseen", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_cls_unseen", avg_acc_r1_coarse, self.progress['num_updates'])
                else:
                    self.writer.add_scalar("acc_coarse_cls", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_cls", avg_acc_r1_coarse, self.progress['num_updates'])
                logger.info("Using [ClS] for Coarse Retrieval Accuracy:")
                logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls_cls['A_r100'], I_r100=recalls_cls['I_r100'], r100_ave=(recalls_cls['A_r100']+recalls_cls['I_r100'])/2, N=N_examples))
                logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls_cls['A_r10'], I_r10=recalls_cls['I_r10'], r10_ave=(recalls_cls['A_r10']+recalls_cls['I_r10'])/2, N=N_examples))
                logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls_cls['A_r5'], I_r5=recalls_cls['I_r5'], r5_ave=(recalls_cls['A_r5']+recalls_cls['I_r5'])/2, N=N_examples))
                logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls_cls['A_r1'], I_r1=recalls_cls['I_r1'], ave_r1=(recalls_cls['A_r1']+recalls_cls['I_r1'])/2,  N=N_examples))
                logger.info(f"validation time: {time.time() - start_val_time:.3f}")
            else:
                recalls_cls = None
            if self.args.feat_matching_weight > 0:
                coarse_cross_relationship_score_matrix = img_feats_list @ audio_feats_total.transpose(0,1)
                recalls_feat = calc_recalls_from_S_one_to_many_coarse(coarse_cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)
                avg_acc_coarse = (recalls_feat['A_r10'] + recalls_feat['I_r10']) / 2
                avg_acc_r1_coarse = (recalls_feat['A_r1'] + recalls_feat['I_r1']) / 2
                if unseen:
                    logger.info("UNSEEN UNSEEN UNSEEN")
                    self.writer.add_scalar("acc_coarse_feat_unseen", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_feat_unseen", avg_acc_r1_coarse, self.progress['num_updates'])
                else:
                    self.writer.add_scalar("acc_coarse_feat", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_feat", avg_acc_r1_coarse, self.progress['num_updates'])
                logger.info("Using mean-pooled feat for Coarse Retrieval Accuracy:")
                logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls_feat['A_r100'], I_r100=recalls_feat['I_r100'], r100_ave=(recalls_feat['A_r100']+recalls_feat['I_r100'])/2, N=N_examples))
                logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls_feat['A_r10'], I_r10=recalls_feat['I_r10'], r10_ave=(recalls_feat['A_r10']+recalls_feat['I_r10'])/2, N=N_examples))
                logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls_feat['A_r5'], I_r5=recalls_feat['I_r5'], r5_ave=(recalls_feat['A_r5']+recalls_feat['I_r5'])/2, N=N_examples))
                logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls_feat['A_r1'], I_r1=recalls_feat['I_r1'], ave_r1=(recalls_feat['A_r1']+recalls_feat['I_r1'])/2,  N=N_examples))
                logger.info(f"validation time: {time.time() - start_val_time:.3f}")
            else:
                recalls_feat = None
        count = 0
        avg_acc_r10 = 0.
        avg_acc_r5 = 0.
        avg_acc_r1 = 0.
        if recalls_cls is not None:
            avg_acc_r10 += (recalls_cls['A_r10'] + recalls_cls['I_r10'])
            avg_acc_r5 += (recalls_cls['A_r5'] + recalls_cls['I_r5'])
            avg_acc_r1 += (recalls_cls['A_r1'] + recalls_cls['I_r1'])
            count += 2

        if recalls_feat is not None and recalls_cls is None: # since cls retrieval is always better, if using cls feat, calculate retrieval based only on it.
            avg_acc_r10 += (recalls_feat['A_r10'] + recalls_feat['I_r10'])
            avg_acc_r5 += (recalls_feat['A_r5'] + recalls_feat['I_r5'])
            avg_acc_r1 += (recalls_feat['A_r1'] + recalls_feat['I_r1'])
            count += 2

        avg_acc_r10 = avg_acc_r10 / count
        avg_acc_r5 = avg_acc_r5 / count
        avg_acc_r1 = avg_acc_r1 / count
        if unseen:
            self.writer.add_scalar("acc_r10_unseen", avg_acc_r10, self.progress['num_updates'])
            self.writer.add_scalar("acc_r5_unseen", avg_acc_r5, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1_unseen", avg_acc_r1, self.progress['num_updates'])
        else:
            self.writer.add_scalar("acc_r10", avg_acc_r10, self.progress['num_updates'])
            self.writer.add_scalar("acc_r5", avg_acc_r5, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1", avg_acc_r1, self.progress['num_updates'])
        return avg_acc_r10, avg_acc_r5, avg_acc_r1

    def _setup_meters(self):
        meters = {}
        if self.args.phase == "pretraining":
            meter_names = ['bce_loss', 'crf_loss', 'data_time', 'train_time']
        else:
            meter_names = ['weighted_loss', 'bce_loss', 'crf_loss', "cls_matching_loss", "feat_matching_loss", "cls_policy_gradient_loss", "feat_policy_gradient_loss", 'data_time', 'train_time']
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
    
    def _setup_models(self):
        if self.args.phase in ['train', 'validate'] and self.args.forceAligned_train:
            model_seg = torch.nn.Identity()
        else:
            model_seg = segmenter.Segmenter(self.args)
        if self.args.phase not in ["pretrain","validate_seg"]:
            model_emb = embedder.Embedder(self.args)
        else:
            model_emb = torch.nn.Identity()

        logger.info(model_seg)
        logger.info(model_emb)
        logger.info("segmenter parameters")
        print_model_info(model_seg)
        logger.info("embedder parameters")
        print_model_info(model_emb)

        
        if self.args.phase in ["validate", "validate_seg"]:
            bundle = torch.load(os.path.join(self.args.exp_dir, "best_bundle.pth"))
            model_seg.load_state_dict(bundle['model_seg'])
            if type(model_emb) != nn.Identity:
                model_emb.load_state_dict(bundle['model_emb'])
            indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info("Perform Validation")
        elif self.progress['num_updates'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"))
            model_seg.load_state_dict(bundle['model_seg'])
            model_emb.load_state_dict(bundle['model_emb'])
            indices = bundle['indices']
            optim_states = bundle['optimizer']
            logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
        else:
            indices = None
            optim_states = None

        if self.args.load_weights_from != None and self.progress['num_updates'] <= 1 and self.args.phase == "pretrain":
            sd = torch.load(self.args.load_weights_from)['model']
            model_seg.carefully_load_state_dict(sd)
        
        if self.args.load_pretrained_segmenter_weights_from != None and self.progress['num_updates'] <= 1 and self.args.phase == "train" and type(model_seg) != torch.nn.Identity:
            sd = torch.load(self.args.load_pretrained_segmenter_weights_from)['model_seg']
            model_seg.load_state_dict(sd)
        
        ## load wavlm weights for the embedder
        if self.args.load_awe_weights_from != None and self.progress['num_updates'] <= 1 and self.args.phase == "train":
            sd = torch.load(self.args.load_awe_weights_from)['model']
            model_emb.awe.carefully_load_state_dict(sd)

        if self.args.feature_grad_mult <= 0.:
            for name, p in model_seg.named_parameters():
                if "feature_extractor" in name:
                    p.requires_grad = False
        if self.args.awe_freeze and type(model_emb) != nn.Identity:
            for p in model_emb.awe.parameters():
                p.requires_grad = False


        trainables = {"seg": [p for p in model_seg.parameters() if p.requires_grad], "emb": [p for p in model_emb.parameters() if p.requires_grad]}

        model_seg.to(self.device)
        model_emb.to(self.device)

        return model_seg, model_emb, trainables, indices, optim_states

    def _setup_dataloader(self):
        train_dataset = spokencoco_dataset.SegEmbDataset(self.args, split='train')
        val_dataset = spokencoco_dataset.SegEmbDataset(self.args, split='val')
        train_sampler = StatefulSampler(len(train_dataset))
        if self.progress['num_updates'] > 1 and self.indices is not None:
            train_sampler.load_state_dict(self.indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
        valid_loader2 = None
        # if self.args.phase == "pretrain":
        #     train_dataset = spokencoco_dataset.SegEmbDataset(self.args, split='train')
        #     val_dataset = spokencoco_dataset.SegEmbDataset(self.args, split='val')
        #     train_sampler = StatefulSampler(len(train_dataset))
        #     if self.progress['num_updates'] > 1 and self.indices is not None:
        #         train_sampler.load_state_dict(self.indices)
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate)
        #     valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
        #     valid_loader2 = None
        # else:
        #     if self.args.places:
        #         # raise NotImplementedError
        #         train_dataset = places_dataset.ImageCaptionDataset(self.args, split='train')
        #         val_seen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_seen')
        #         val_unseen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_unseen')
        #         train_sampler = StatefulSampler(len(train_dataset))
        #         if self.progress['num_updates'] > 1 and self.indices is not None:
        #             train_sampler.load_state_dict(self.indices)
        #         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
        #         valid_loader = torch.utils.data.DataLoader(val_seen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_seen_dataset.collate)
        #         valid_loader2 = torch.utils.data.DataLoader(val_unseen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_unseen_dataset.collate)
        #     else:
        #     # SpokenCOCO
        #         train_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='train')
        #         val_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='val')
        #         train_sampler = StatefulSampler(len(train_dataset))
        #         if self.progress['num_updates'] > 1 and self.indices is not None:
        #             train_sampler.load_state_dict(self.indices)
        #         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=False)
        #         valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
        #         valid_loader2 = None

        return train_loader, valid_loader, valid_loader2, train_sampler, len(train_dataset)
    
    def _setup_optimizer(self):
        optimizer_seg = BertAdam(self.trainables['seg'], lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates//2 - self.args.freeze_segmenter)

        if self.progress['num_updates'] > 1:
            optimizer_seg.load_state_dict(self.optim_states['seg'])
            for state in optimizer_seg.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer_seg.zero_grad()

        optimizer_emb = BertAdam(self.trainables['emb'], lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates//2)

        if self.progress['num_updates'] > 1:
            optimizer_emb.load_state_dict(self.optim_states['emb'])
            for state in optimizer_emb.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer_emb.zero_grad()
        return {"seg": optimizer_seg, "emb": optimizer_emb}
    
    def _setup_scheduler(self):
        pass

    def weight_loss(self, losses):
        weighted_loss = 0.
        # print(losses)
        # if self.args.phase == "pretrain" or self.progress['num_updates'] > self.args.freeze_segmenter:
        if "bce_loss" in losses:
            weighted_loss += losses['bce_loss'] * self.args.bce_weight
        if "crf_loss" in losses:
            weighted_loss += losses['crf_loss'] * self.args.crf_weight
        if "cls_policy_gradient_loss" in losses:
            weighted_loss += losses['cls_policy_gradient_loss'] * self.args.cls_policy_gradient_weight
        if "feat_policy_gradient_loss" in losses:
            weighted_loss += losses['feat_policy_gradient_loss'] * self.args.feat_policy_gradient_weight
        if "cls_matching_loss" in losses:
            weighted_loss += losses['cls_matching_loss'] * self.args.cls_matching_weight
        if "feat_matching_loss" in losses:
            weighted_loss += losses['feat_matching_loss'] * self.args.feat_matching_weight
        
        return weighted_loss
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

