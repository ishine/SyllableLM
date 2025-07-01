## SegmentalSSE

This is where I (Alan) trained the Sylboost iterations.
This uses Puyuan Peng's Segmental SSE code to additionally validate training against ground truth boundaries
To implement this, I hacked on model loading (models/segmenter/Segmenter2) that loads either hubert or d2v2

The dataloading really just needs an absolute path to librispeech in datasets/spokencoco_dataset.py
You will also want to load in boundary pseudolabels for each file loaded

The real logic is in Segmenter2, which takes in onehot indices at the boundaries of pseudo syllabic units at the unit sequence length (for Hubert/data2vec2 this is 50Hz). The onehot index should be after an instantaneous boundary
(So if the boundary is at 0.02s, then the value at index 1 (0-indexed) should be 1)

If you can load librispeech files and indices into the forward pass you don't need the data loading code.
This model converges extremely quickly