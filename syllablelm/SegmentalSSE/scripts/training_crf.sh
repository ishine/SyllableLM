#!/bin/sh
#source ~/miniconda3/etc/profile.d/conda.sh

# bash training_crf.sh alan_a40 hubert_base_ls960.pt 3 0 0 pretrain 0.95 image 1 20 1 9 0.00001 -1000 4 0 30 none 0 1 0 0 0 0


machine=$1
pretrained_model=$2 # WavLM-Base.pt, hubert_base_ls960.pt, wav2vec_small.pt
positive_weight=$3
use_fa_tgt=$4
resume=$5 # 0 or 1
phase=$6 # pretrain, train, validate
seg_quantile_threshold=$7

anchor_type=$8
use_audio_cls_token=$9
max_num_words=${10}
combiner_encoder_layers=${11}
awe_encoder_layers=${12}
cls_policy_gradient_weight=${13}
feat_policy_gradient_weight=${14}
cls_matching_weight=${15}
feat_matching_weight=${16}
seg_cap=${17}
load_pretrained_segmenter_weights_from=${18}
forceAligned_train=${19}
crf=1
crf_infer=1
crf_weight=${20}
bce_weight=${21}
freeze_segmenter=${22}
running_average=${23}
awe_freeze=${24}

if [ "$phase" = "validate_seg" ]; then
    phase_fn="pretrain"
elif [ "$phase" = "validate" ]; then
    phase_fn="train"
else
    phase_fn=$phase
fi

exp_name=${machine}_runAve${running_average}_crf${crf}_crfWei${crf_weight}_bceWei${bce_weight}_freezeSeg${freeze_segmenter}_${phase_fn}_faTrain${forceAligned_train}_posWeight${positive_weight}_useFa${use_fa_tgt}_segQuan${seg_quantile_threshold}_anc${anchor_type}_cls${use_audio_cls_token}_comLayer${combiner_encoder_layers}_$aweLayer${awe_encoder_layers}_clsPolicy${cls_policy_gradient_weight}_featPolicy${feat_policy_gradient_weight}_clsMatch${cls_matching_weight}_featMatch${feat_matching_weight}_aweFreeze${awe_freeze}


mkdir -p ./logs


if [ "${machine}" = "alan_a40" ]; then
    pretrained_root=/data/scratch/abaade/hubert
    exp_root=/data/scratch/abaade/cluster/SegmentalSSE/exp # i.e. the parent folder of the exp folder
    data_root=/data/scratch/pyp/datasets/coco_pyp
    boundary_root=/saltpool0/scratch/pyp/discovery/word_unit_discovery
    # conda activate fairseq_torch2
fi

if [ "$phase" = "train" ]; then
    batch_size=100
    n_epochs=20
else
    batch_size=32
    n_epochs=3
fi


python \
../run_spokencoco.py \
--running_average ${running_average} \
--freeze_segmenter ${freeze_segmenter} \
--crf ${crf} \
--crf_infer ${crf_infer} \
--crf_weight ${crf_weight} \
--bce_weight ${bce_weight} \
--forceAligned_train ${forceAligned_train} \
--awe_freeze ${awe_freeze} \
--anchor_type ${anchor_type} \
--use_audio_cls_token ${use_audio_cls_token} \
--max_num_words ${max_num_words} \
--combiner_encoder_layers ${combiner_encoder_layers} \
--awe_encoder_layers ${awe_encoder_layers} \
--cls_policy_gradient_weight ${cls_policy_gradient_weight} \
--feat_policy_gradient_weight ${feat_policy_gradient_weight} \
--cls_matching_weight ${cls_matching_weight} \
--feat_matching_weight ${feat_matching_weight} \
--seg_cap ${seg_cap} \
--seg_quantile_threshold ${seg_quantile_threshold} \
--resume $resume \
--use_fa_tgt ${use_fa_tgt} \
--positive_weight ${positive_weight} \
--seg_threshold 0.5 \
--boundary_tolerance 0.04 \
--phase ${phase} \
--random_init_last_x 0 \
--load_weights_from ${pretrained_root}/${pretrained_model} \
--load_awe_weights_from ${pretrained_root}/${pretrained_model} \
--load_pretrained_segmenter_weights_from ${load_pretrained_segmenter_weights_from} \
--opt_level O0 \
--audio_feat_len 8 \
--val_audio_feat_len 8 \
--num_workers 4 \
--train_audio_dataset_json_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_ls_iter_hubertbase_iter2_w00675_q75/train.json \
--val_audio_dataset_json_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_ls_iter_hubertbase_iter2_w00675_q75/val.json \
--train_boundary_pkl_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_ls_iter_hubertbase_iter2_w00675_q75/all_boundaries_10.pkl \
--val_boundary_pkl_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_ls_iter_hubertbase_iter2_w00675_q75/all_boundaries_10.pkl \
--exp_dir ${exp_root}/${exp_name} \
--batch_size ${batch_size} \
--val_batch_size 16 \
--n_epochs ${n_epochs} \
--n_print_steps 25 \
--n_val_steps 625 \
--lr 0.00005 \
--warmup_fraction 0.1 \
--feature_grad_mult 0. \
--encoder_layers 12 \
--layer_use 11


#python \
#../run_spokencoco.py \
#--running_average ${running_average} \
#--freeze_segmenter ${freeze_segmenter} \
#--crf ${crf} \
#--crf_infer ${crf_infer} \
#--crf_weight ${crf_weight} \
#--bce_weight ${bce_weight} \
#--forceAligned_train ${forceAligned_train} \
#--awe_freeze ${awe_freeze} \
#--anchor_type ${anchor_type} \
#--use_audio_cls_token ${use_audio_cls_token} \
#--max_num_words ${max_num_words} \
#--combiner_encoder_layers ${combiner_encoder_layers} \
#--awe_encoder_layers ${awe_encoder_layers} \
#--cls_policy_gradient_weight ${cls_policy_gradient_weight} \
#--feat_policy_gradient_weight ${feat_policy_gradient_weight} \
#--cls_matching_weight ${cls_matching_weight} \
#--feat_matching_weight ${feat_matching_weight} \
#--seg_cap ${seg_cap} \
#--seg_quantile_threshold ${seg_quantile_threshold} \
#--resume $resume \
#--use_fa_tgt ${use_fa_tgt} \
#--positive_weight ${positive_weight} \
#--seg_threshold 0.5 \
#--boundary_tolerance 0.04 \
#--phase ${phase} \
#--random_init_last_x 2 \
#--load_weights_from ${pretrained_root}/${pretrained_model} \
#--load_awe_weights_from ${pretrained_root}/${pretrained_model} \
#--load_pretrained_segmenter_weights_from ${load_pretrained_segmenter_weights_from} \
#--opt_level O0 \
#--audio_feat_len 8 \
#--val_audio_feat_len 8 \
#--num_workers 4 \
#--train_audio_dataset_json_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_coco/train.json \
#--val_audio_dataset_json_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_coco/val.json \
#--train_boundary_pkl_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_coco/all_boundaries.pkl \
#--val_boundary_pkl_file /data/scratch/abaade/cluster/SegmentalSSE/pseudo_coco/all_boundaries.pkl \
#--exp_dir ${exp_root}/${exp_name} \
#--batch_size ${batch_size} \
#--val_batch_size 50 \
#--n_epochs ${n_epochs} \
#--n_print_steps 25 \
#--n_val_steps 250 \
#--lr 0.00005 \
#--warmup_fraction 0.1 \
#--feature_grad_mult 0. \
#--encoder_layers 12 \
#--layer_use 11 # >> "./logs/${exp_name}.log" 2>&1
# --layer_use 11


#--train_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy_with_alignments.json \
#--val_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json \
#--train_boundary_pkl_file ${boundary_root}/mbr_104_1030_top10/train_data_dict.pkl  \
#--val_boundary_pkl_file ${boundary_root}/mbr_104_1030_top10/val_data_dict.pkl  \


# python \
# ../run_spokencoco.py \
# --freeze_segmenter ${freeze_segmenter} \
# --crf ${crf} \
# --crf_infer ${crf_infer} \
# --crf_weight ${crf_weight} \
# --bce_weight ${bce_weight} \
# --forceAligned_train ${forceAligned_train} \
# --awe_freeze 1 \
# --anchor_type ${anchor_type} \
# --use_audio_cls_token ${use_audio_cls_token} \
# --max_num_words ${max_num_words} \
# --combiner_encoder_layers ${combiner_encoder_layers} \
# --awe_encoder_layers ${awe_encoder_layers} \
# --cls_policy_gradient_weight ${cls_policy_gradient_weight} \
# --feat_policy_gradient_weight ${feat_policy_gradient_weight} \
# --cls_matching_weight ${cls_matching_weight} \
# --feat_matching_weight ${feat_matching_weight} \
# --seg_cap ${seg_cap} \
# --seg_quantile_threshold ${seg_quantile_threshold} \
# --resume $resume \
# --use_fa_tgt ${use_fa_tgt} \
# --positive_weight ${positive_weight} \
# --seg_threshold 0.5 \
# --boundary_tolerance 0.04 \
# --phase ${phase} \
# --random_init_last_x 3 \
# --load_weights_from ${pretrained_root}/${pretrained_model} \
# --load_awe_weights_from ${pretrained_root}/${pretrained_model} \
# --load_pretrained_segmenter_weights_from ${load_pretrained_segmenter_weights_from} \
# --opt_level O1 \
# --audio_feat_len 8 \
# --val_audio_feat_len 12 \
# --num_workers 4 \
# --train_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy_with_alignments.json \
# --val_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json \
# --train_boundary_pkl_file /saltpool0/scratch/pyp/discovery/word_unit_discovery/mbr_104_1030_top10/train_data_dict.pkl  \
# --val_boundary_pkl_file /saltpool0/scratch/pyp/discovery/word_unit_discovery/mbr_104_1030_top10/val_data_dict.pkl  \
# --exp_dir ${exp_root}/${exp_name} \
# --batch_size ${batch_size} \
# --val_batch_size 100 \
# --n_epochs ${n_epochs} \
# --n_print_steps 200 \
# --n_val_steps 400 \
# --lr 0.00005 \
# --warmup_fraction 0.1 \
# --feature_grad_mult 0. \
# --encoder_layers 12 \
# --layer_use 11 >> "./logs/${exp_name}.log" 2>&1
# # --layer_use 11

# python \
# ../run_spokencoco.py \
# --freeze_segmenter ${freeze_segmenter} \
# --crf ${crf} \
# --crf_infer ${crf_infer} \
# --crf_weight ${crf_weight} \
# --bce_weight ${bce_weight} \
# --forceAligned_train ${forceAligned_train} \
# --awe_freeze 1 \
# --anchor_type ${anchor_type} \
# --use_audio_cls_token ${use_audio_cls_token} \
# --max_num_words ${max_num_words} \
# --combiner_encoder_layers ${combiner_encoder_layers} \
# --awe_encoder_layers ${awe_encoder_layers} \
# --cls_policy_gradient_weight ${cls_policy_gradient_weight} \
# --feat_policy_gradient_weight ${feat_policy_gradient_weight} \
# --cls_matching_weight ${cls_matching_weight} \
# --feat_matching_weight ${feat_matching_weight} \
# --seg_cap ${seg_cap} \
# --seg_quantile_threshold ${seg_quantile_threshold} \
# --resume $resume \
# --use_fa_tgt ${use_fa_tgt} \
# --positive_weight ${positive_weight} \
# --seg_threshold 0.5 \
# --boundary_tolerance 2 \
# --phase ${phase} \
# --random_init_last_x 3 \
# --load_weights_from ${pretrained_root}/${pretrained_model} \
# --load_awe_weights_from ${pretrained_root}/${pretrained_model} \
# --load_pretrained_segmenter_weights_from ${load_pretrained_segmenter_weights_from} \
# --opt_level O1 \
# --audio_feat_len 8 \
# --val_audio_feat_len 12 \
# --num_workers 4 \
# --train_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy_with_alignments.json \
# --val_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json \
# --train_boundary_pkl_file /saltpool0/scratch/pyp/discovery/word_unit_discovery/mbr_104_1030_top10/train_data_dict.pkl  \
# --val_boundary_pkl_file /saltpool0/scratch/pyp/discovery/word_unit_discovery/mbr_104_1030_top10/val_data_dict.pkl  \
# --exp_dir ${exp_root}/${exp_name} \
# --batch_size ${batch_size} \
# --val_batch_size 100 \
# --n_epochs ${n_epochs} \
# --n_print_steps 200 \
# --n_val_steps 400 \
# --lr 0.00005 \
# --warmup_fraction 0.1 \
# --feature_grad_mult 0. \
# --encoder_layers 12 \
# --layer_use 11 >> "./logs/${exp_name}.log" 2>&1
# # --layer_use 11