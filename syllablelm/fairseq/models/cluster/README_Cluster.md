### Main model training

python fairseq_cli/hydra_train.py \
--config-dir  \
fairseq/models/cluster/config \
--config-name \
cluster_alan \
task.data=/data/scratch/abaade/cluster/tsv/libri_light_bulk_d2v2/ \
common.log_interval=100 \
dataset.max_tokens=160 \
distributed_training.distributed_world_size=2 \
model.vocab_size=8192 \
task.vocab_size=8192 \
+model.depth=12 \
+model.num_heads=12 \
+model.embed_dim=768 \
dataset.disable_validation=True \
optimization.lr=[0.0002] \
+task.reducer_path="/saltpool0/scratch/abaade/cluster/LibriSpeech_bulk_2116_24576_agg_8192.npy" \
+task.tsv_replace_source="LibriSpeech_bulk_2120_32768_w23_0019" \
+task.tsv_replace_target="LibriSpeech_bulk_2116_24576" \
hydra.run.dir="/data/scratch/abaade/cluster/models/syl_d2v2_LibriSpeech_bulk_2116_24576_agg_8192_equal_compute_1"


### Interleaved model training

python fairseq_cli/hydra_train.py \
--config-dir \
fairseq/models/cluster/config \
--config-name \
cluster_alan_interleaved \
task.data=/data/scratch/abaade/cluster/tsv/libri_light_full_interleave/ \
common.log_interval=100 \
task.interleave_strat=5 \
dataset.max_tokens=128 \
+model.vocab_size=10243 \
+task.vocab_size_long=8192 \
dataset.disable_validation=True \
distributed_training.distributed_world_size=2 \
optimization.lr=[0.0002] \
optimization.update_freq=[1] \
+task.reducer_path="/saltpool0/scratch/abaade/cluster/LibriSpeech_bulk_2116_24576_agg_8192.npy" \
+task.tsv_replace_source="LibriSpeech_bulk_2120_32768_w23_0019" \
+task.tsv_replace_target="LibriSpeech_bulk_2116_24576" \
+task.interleave_long_grouping=1 \
hydra.run.dir="/data/scratch/abaade/cluster/models/syl_d2v2_interleaved5_LibriSpeech_bulk_2116_24576_8192"

