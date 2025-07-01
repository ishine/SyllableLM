## fairseq folder for SyllableLM

This folder contains the main modeling code and training logic in the unfortunately deprecated fairseq library

This contains only training code and not inference code
Due to the complexity of the interleaved vocoder, I only implemented non-kv cache based sampling from all models
This is contained in a very long jupyter notebook that I am still filtering for public release

The losses provided are sufficient to replicate our results on the Swuggy-style datasets, however.

The setup is standard fairseq. The transformer code is self-contained and somewhat located in the valle/ folder

Training commands are in README_Cluster