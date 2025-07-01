import argparse

def MyParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # customized
    parser.add_argument("--alternate", type=int, default=0, help='if 1, optimize segmenter and embedder in a alternating way')
    parser.add_argument("--running_average", type=int, default=0, help="keep a running average of the negative reward, and the negative reward used in calculation will be the negative reward minus this running average. specify 1 if we want to use running average")
    parser.add_argument("--freeze_segmenter", type=int, default=0, help="the number of epochs where we do not update parameters of segmenter in the train phase")
    parser.add_argument("--crf", type=int, default=0, help="wether or not use Linear Chain CRF in the segmenter")
    parser.add_argument("--crf_infer", type=int, default=1, help='also show result of crf inference')
    parser.add_argument("--forceAligned_train", type=int, default=0, help="if 1, that means we use force aligned boundary for training, and therefore no need to have segmenter")
    parser.add_argument("--anchor_type", type=str, default="image", help="text or image as the anchor, pre-extracted")
    parser.add_argument("--use_audio_cls_token", type=int, default=1)
    parser.add_argument("--max_num_words", type=int, default=30, help="maximal number of segments per utterance, used for combiner padding")
    parser.add_argument("--combiner_encoder_layers", type=int, default=1, help="as light as possible")
    parser.add_argument("--awe_encoder_layers", type=int, default=9, help="I choose 9 based on figure 4 of Ankita's comparative feature analysis paper: https://arxiv.org/pdf/2211.03929.pdf")
    parser.add_argument("--crf_weight", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--cls_policy_gradient_weight", type=float, default=1.0)
    parser.add_argument("--feat_policy_gradient_weight", type=float, default=1.0)
    parser.add_argument("--cls_matching_weight", type=float, default=1.0)
    parser.add_argument("--feat_matching_weight", type=float, default=1.0)
    parser.add_argument("--seg_cap", type=int, default=30, help="the maximal segment length, use to cap the each segment output from the segmenter. i.e. 30 means the maximal length will be 30 frames")
    parser.add_argument("--seg_quantile_threshold", type=float, default=0, help="use a quantile on the probability to select threshold, rather than a fix seg_threshold for all samples. for example 0.95 means predict frames that has prob above top 5 percent. 0 Means use the fixed seg_threshold")
    parser.add_argument("--use_fa_tgt", type=int, default=0, help="1 if use force aligned boundary as training target, 0 otherwise")
    parser.add_argument("--positive_weight", type=float, default=10., help="since boundary prediction is highly imbalanced, we put more weight on the positive instances of the binary cross entropy formula")
    parser.add_argument("--seg_threshold", type=float, default=0.5, help="threshold on the probability that decides if is a boundary")
    parser.add_argument("--boundary_tolerance", type=float, default=0.02, help="in terms of frame, 1 frame is equivalent to 20ms if using w2v2/hubert/wavlm")
    parser.add_argument("--load_weights_from", type=str, default="/data/scratch/pyp/exp_pyp/discovery/pretrained_models/WavLM-Base.pt", help="for segmenter")
    parser.add_argument("--load_pretrained_segmenter_weights_from", type=str, default="/data/scratch/pyp/exp_pyp/SegmentalSSE/exp1/best_bundle.pth", help="for segmenter, load the model from *my boundary prediction pretraining*")
    parser.add_argument("--load_awe_weights_from", type=str, default="/data/scratch/pyp/exp_pyp/discovery/pretrained_models/WavLM-Base.pt", help="for Net1 in the Embedder")
    parser.add_argument("--awe_freeze", type=int, default=1, help="freeze the AWE model (i.e. Net1 in the embedder) during training, 0 means not freezing it")

    # outter most params
    parser.add_argument("--resume", type=int, default=0, help="load from exp_dir if 1")
    parser.add_argument("--phase", default="pretrain", type=str, help="'pretrain' means segmenter pretraining using vghubert segmentaition results, 'train' means segmenter+embedder contrastive training with BERT embedding, or CLIP embedding of text or image, 'validate' means test the segmentation result of the segmenter and the retrieval result of segmenter+embedder", choices=['pretrain', 'train', 'validate', 'validate_seg'])

    # dataset args
    parser.add_argument("--spf", default=0.02, type=float, help="second per frame, 0.02 for w2v2, hubert, wavlm")
    parser.add_argument("--downsample_factor", default=320, type=int, help="same for w2v2 hubert wavlm")
    parser.add_argument("--train_boundary_pkl_file", type=str, default="/saltpool0/scratch/pyp/discovery/word_unit_discovery/mbr_104_1030_top10/train_data_dict.pkl")
    parser.add_argument("--val_boundary_pkl_file", type=str, default="/saltpool0/scratch/pyp/discovery/word_unit_discovery/mbr_104_1030_top10/val_data_dict.pkl")
    parser.add_argument("--train_audio_dataset_json_file", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy_with_alignments.json")
    parser.add_argument("--val_audio_dataset_json_file", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json")
    parser.add_argument("--audio_feat_len", type=float, help="maximal audio length", default=8)
    parser.add_argument("--val_audio_feat_len", type=float, help="maximal audio length", default=10.)
    parser.add_argument("--normalize", action="store_true", default=False, help="whether or not normalize raw input, both w2v2 and hubert base doesn't normalize the input, but in exps in two papers, we normalized it, hopefully this doesn't make a difference")
    
    # trainer args
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--val_cross_batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--n_print_steps", type=int)
    parser.add_argument("--n_val_steps", type=int)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--opt_level", type=str, default="O0", help="O0, O1, O2, O3. O0:fp32, O1:fp16+fp32 mixed, O2:almost fp16, O3:fp16")
    


    # segmenter arguments
    parser.add_argument(
        "--extractor_mode",
        choices=["default", "layer_norm"],
        help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
        default="default"
    )

    parser.add_argument(
        "--encoder_layers",
        type=int,
        metavar="L",
        help="num encoder layers in the transformer",
        default=12
    )
    parser.add_argument(
        "--encoder_embed_dim",
        type=int,
        metavar="H",
        help="encoder embedding dimension",
        default=768
    )
    parser.add_argument(
        "--encoder_ffn_embed_dim",
        type=int,
        metavar="F",
        help="encoder embedding dimension for FFN",
        default=3072
    )
    parser.add_argument(
        "--encoder_attention_heads",
        type=int,
        metavar="A",
        help="num encoder attention heads",
        default=12
    )
    parser.add_argument(
        "--activation-fn",
        help="activation function to use",
        default="gelu"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        metavar="D",
        help="dropout probability for the transformer",
        default=0.1
    )

    parser.add_argument(
        "--attention_dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights",
        default=0.1
    )

    parser.add_argument(
        "--activation_dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN",
        default=0.0
    )

    parser.add_argument(
        "--final_dim",
        type=int,
        metavar="D",
        help="project final representations and targets to this many dimensions",
        default=256 # 0 means use encoder_embed_dim as the final_dim
    )

    parser.add_argument(
        "--layer_norm_first",
        action="store_true",
        help="apply layernorm first in the transformer",
        default=False
    )

    parser.add_argument(
        "--encoder_layerdrop",
        type=float,
        help="probability of dropping a tarnsformer layer",
        default=0.0
    )

    parser.add_argument(
        "--conv_feature_layers",
        type=str,
        metavar="EXPR",
        help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        default= "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
    )

    parser.add_argument(
        "--logit_temp", type=float, help="temperature to divide logits by", default=0.1
    )

    parser.add_argument(
        "--quantize_targets", action="store_true", help="use the original convnet feature as input to the transformer, quantized convnet features that are reconstruction targets when calculating the loss (w2v2 paper)", default=True
    )

    parser.add_argument(
        "--quantize_input", action="store_true", help="quantized all outputs of convnet feature extractor and use them as the input to the transformer", default=False
    )

    parser.add_argument(
        "--same_quantizer", action="store_true", help="use the same quantizer to quantize both transformer input and output", default=False
    )
    parser.add_argument(
        "--feature_grad_mult",
        type=float,
        help="multiply feature extractor var grads by this",
        default=0.1 # the paper use 0.1 i.e. scale down the gradient by a factor of 10
    )

    parser.add_argument(
        "--latent_vars",
        type=int,
        metavar="N",
        help="number of latent variables V in each group of the codebook",
        default=320
    )

    parser.add_argument(
        "--latent_groups",
        type=int,
        metavar="N",
        help="number of groups G of latent variables in the codebook",
        default=2
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        metavar="N",
        help="if > 0, uses this dimensionality for latent variables (code dimension). otherwise uses final_dim / latent_groups",
        default=0
    )

    parser.add_argument("--mask_length", type=int, help="mask length", default=10)

    parser.add_argument( "--mask_prob", type=float, help="probability of replacing a token with mask, default is 0.65, since mask_length=10, this is equivalent to all the tokens in a sequence has 0.065 probability to be starting tokens of a masked span", default=0.65)

    parser.add_argument(
        "--mask_selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
        default="static"
    )

    parser.add_argument(
        "--mask_other",
        type=float,
        help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        default=0
    )

    parser.add_argument(
        "--no_mask_overlap",
        action="store_true",
        help="whether to allow masks to overlap",
        default=False
    )

    parser.add_argument(
        "--mask_min_space",
        type=int,
        help="min space between spans (if no overlap is enabled)",
        default=1
    )

    parser.add_argument(
        "--mask_channel_length",
        type=int,
        help="repeat the mask indices multiple times",
        default=0.0
    )

    parser.add_argument(
        "--mask_channel_prob",
        type=float,
        help="probability of replacing a token with mask",
        default=0.0
    )

    parser.add_argument(
        "--mask_channel_selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
        default="static"
    )

    parser.add_argument(
        "--mask_channel_other",
        type=float,
        help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        default=0
    )

    parser.add_argument(
        "--no_mask_channel_overlap",
        action="store_true",
        help="whether to allow masks to overlap",
        default=False
    )

    parser.add_argument(
        "--mask_channel_min_space",
        type=int,
        help="min space between spans (if no overlap is enabled)",
        default=1
    )

    parser.add_argument(
        "--dropout_input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
        default=0.1
    )

    parser.add_argument(
        "--dropout_features",
        type=float,
        metavar="D",
        help="dropout to apply to the features (after feat extr)",
        default=0.1
    )

    parser.add_argument(
        "--num_negatives", type=int, metavar="N", help="number of negative examples",
        default=100
    )

    parser.add_argument(
        "--negatives_from_everywhere",
        action="store_true",
        help="sample negatives from everywhere, not just masked states",
        default=False
    )

    parser.add_argument(
        "--cross_sample_negatives",
        type=int,
        metavar="N",
        help="num of cross sampled negatives",
        default=0
    )

    parser.add_argument(
        "--codebook_negatives",
        type=int,
        metavar="N",
        help="num of codebook sampled negatives",
        default=0
    )

    parser.add_argument(
        "--conv_pos",
        type=int,
        metavar="N",
        help="kernel size for convolutional positional embeddings",
        default=128
    )

    parser.add_argument(
        "--conv_pos_groups",
        type=int,
        metavar="N",
        help="number of groups for convolutional positional embedding",
        default=16
    )

    parser.add_argument(
        "--latent_temp",
        type=str,
        metavar="D",
        help="temperature for latent variable sampling. can be string of tuple of 3 values (start, end, decay)",
        default="(2,0.5,0.999995)"
    )

    parser.add_argument(
        "--target_glu", action="store_true", help="adds projection + glu to targets", default=False
    )

    parser.add_argument(
        "--conv-bias", action="store_true", help="include bias in conv encoder", default=False
    )

    parser.add_argument(
        "--layer_use", type=int, help="which layer feat to use to input to second tranformer, range from 0 to encoder_layer - 1", default=4
    )
    
    parser.add_argument(
        "--diversity_weight", type=float, help="weight on the diversity loss", default=0.1
    )

    parser.add_argument(
        "--return_code_index", action="store_true", default=False, help="return the code index"
    )
    
    
    parser.add_argument(
        "--trim_mask", action="store_true", default=False
    )
    
    # hubert args
    parser.add_argument("--untie_final_proj", type = bool, default=True,
        help = "use separate projection for each target"
    )
    parser.add_argument("--skip_masked", type = bool, default=False,
        help = "skip computing losses over masked frames"
    )
    parser.add_argument("--skip_nomask", type = bool, default=False,
        help = "skip computing losses over unmasked frames"
    )
    parser.add_argument(
        "--pred_masked_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--pred_nomask_weight", type=float, default=0.0
    )



    parser.add_argument(
        "--random_init_last_x", type=int, default=None
    )
    
    parser.add_argument(
        "--freeze_first_x", type=int, default=None
    )


    return parser