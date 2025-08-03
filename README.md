## Official Implementations "Faster Diffusion: Rethinking the Role of the Encoder for Diffusion Model Inference" for DiT (NeurIPS'24)

[![arXiv](https://img.shields.io/badge/arXiv-FasterDiffusion-<COLOR>.svg)](https://arxiv.org/abs/2312.09608) [![arXiv](https://img.shields.io/badge/paper-FasterDiffusion-b31b1b.svg)](https://arxiv.org/abs/2312.09608.pdf) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=sen-mao/DiT-FasterDiffusion)


![example](visuals/example.gif)

## Requirements

First, download and set up the repo:

```bash
git clone https://github.com/sen-mao/DiT-FasterDiffusion.git
cd DiT-FasterDiffusion
```

A suitable conda environment named `DiT-FasterDiffusion` can be created
and activated with:

```
conda env create -f environment.yml

conda activate DiT-FasterDiffusion
```

## Sampling

### Run the `sample_fasterdiffusion.py` to generate images for DiT.
```bash
python sample_fasterdiffusion.py --only-DiT 1 
```


![DiT samples](visuals/infer_dit.png)

### Run the `sample_fasterdiffusion.py` to generate images for DiT with FasterDiffusion.
```bash
python sample_fasterdiffusion.py 
```

![DiT_FasterDiffusion samples](visuals/infer_dit_fasterdiffusion.png)

## Evaluation
DiT provides a script for evaluation [sample_ddp.py](https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py).
The evaluation code is modified from this code, and the 50k sampling results are saved in the same data format as [ADM](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py).
The evaluation code is obtained from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations), and the evaluation environment is already included in the FasterDiffusion-DiT environment.


- DiT-XL/2-G (cfg=1.50)

```bash
#!/bin/bash

export NCCL_P2P_DISABLE=1

NUM_GPUS=8
BATCH_SIZE=32

echo 'DiT:'
export CFG_SCALE=1.5
MODEL_FLAGS="--model DiT-XL/2 --per-proc-batch-size 64 --num-fid-samples 50000 --image-size 256 --only-DiT True"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py  $MODEL_FLAGS --cfg-scale $CFG_SCALE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz

echo 'DiT (FasterDiffusion):'
export CFG_SCALE=1.5
MODEL_FLAGS="--model DiT-XL/2 --num-fid-samples 50000 --image-size 256"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py $MODEL_FLAGS --cfg-scale $CFG_SCALE --per-proc-batch-size $BATCH_SIZE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz
```



## BibTeX

```bibtex
@article{li2024faster,
  title={Faster diffusion: Rethinking the role of the encoder for diffusion model inference},
  author={Li, Senmao and Hu, Taihang and van de Weijer, Joost and Shahbaz Khan, Fahad and Liu, Tao and Li, Linxuan and Yang, Shiqi and Wang, Yaxing and Cheng, Ming-Ming and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={85203--85240},
  year={2024}
}

@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```


## Acknowledgments
This codebase is built based on original [DiT](https://github.com/facebookresearch/DiT), and reference [MDT](https://github.com/sail-sg/MDT/tree/main) code. Thanks very much.

### Contact
If you have any questions, please feel free to reach out to me at  `senmaonk@gmail.com`. 