## Official Implementations "Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models" for DiT (NeurIPS'24)

[![arXiv](https://img.shields.io/badge/arXiv-FasterDiffusion-<COLOR>.svg)](https://arxiv.org/abs/2312.09608) [![arXiv](https://img.shields.io/badge/paper-FasterDiffusion-b31b1b.svg)](https://arxiv.org/abs/2312.09608.pdf) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=sen-mao/DiT-FasterDiffusion)

[//]: # (> **Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models**)

[//]: # (>)

[//]: # (> [Senmao Li]&#40;https://github.com/sen-mao&#41;, [Taihang Hu]&#40;https://github.com/hutaiHang&#41;, [Joost van de Weijer]&#40;https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en&oi=sra&#41;, [Fahad Khan]&#40;https://sites.google.com/view/fahadkhans/home&#41;, [Tao Liu]&#40;ltolcy0@gmail.com&#41;, [Linxuan Li]&#40;https://github.com/Potato-lover&#41;, [Shiqi Yang]&#40;https://www.shiqiyang.xyz/&#41;, [Yaxing Wang]&#40;https://yaxingwang.netlify.app/author/yaxing-wang/&#41;, [Ming-Ming Cheng]&#40;https://mmcheng.net/&#41;, [Jian Yang]&#40;https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en&#41;)

[//]: # (The official codebase for [FasterDiffusion]&#40;https://arxiv.org/abs/2312.09608&#41; accelerates [DiT]&#40;https://github.com/facebookresearch/DiT&#41; with **~1.51x** speedup.)

[//]: # ()
[//]: # ()
[//]: # (![DiT samples]&#40;visuals/infer_dit.jpg&#41;)

[//]: # ()
[//]: # (Results of DiT &#40;top&#41; and this method in conjunction with our proposed approach &#40;bottom&#41;.)

<div align="center">
  <img src="visuals/example.gif" width="100%" ></img>
  <br>
</div>
<br>

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

## Performance

| DiT Model                                                                                 | Resolution | FID&darr; | sFID&darr; | IS&uarr; | Precision&uarr; | Recall&uarr; | s/image&darr; |
|-------------------------------------------------------------------------------------------|:----------:|:---------:|:----------:|:--------:|:---------------:|:------------:|:-------------:|
| [XL/2-G](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)                   |  256x256   |   2.27    |    4.60    |  278.24  |      0.83       |     0.57     |     5.13      |
| [XL/2-G](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) (FasterDiffusion) |  256x256   |   2.31    |    4.55    |  276.05  |      0.82       |     0.57     |     **3.26**      |


## Sampling

### Run the `sample_fasterdiffusion.py` to generate images for DiT.
```bash
python sample_fasterdiffusion.py --only-DiT 1 
```

<details>
<summary>Output:</summary>

```bash
key time-steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
Warming up GPU...
100%|█████████████████████████████████████████| 250/250 [00:41<00:00,  6.09it/s]
100%|█████████████████████████████████████████| 250/250 [00:40<00:00,  6.16it/s]
DiT: 5.11 seconds/image
```

</details>

![DiT samples](visuals/infer_dit.png)

### Run the `sample_fasterdiffusion.py` to generate images for DiT with FasterDiffusion.
```bash
python sample_fasterdiffusion.py 
```

<details>
<summary>Output:</summary>

```bash
key time-steps = [0, 6, 7, 8, 9, 16, 17, 18, 19, 26, 27, 28, 29, 36, 37, 38, 39, 46, 47, 48, 49, 56, 57, 58, 59, 66, 67, 68, 69, 76, 77, 78, 79, 86, 87, 88, 89, 96, 97, 98, 99, 106, 107, 108, 109, 116, 117, 118, 119, 126, 127, 128, 129, 136, 137, 138, 139, 146, 147, 148, 149, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 186, 187, 188, 189, 196, 197, 198, 199, 206, 207, 208, 209, 216, 217, 218, 219, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
Warming up GPU...
100%|█████████████████████████████████████████| 250/250 [00:26<00:00,  9.32it/s]
100%|█████████████████████████████████████████| 250/250 [00:26<00:00,  9.52it/s]
DiT (FasterDiffusion): 3.29 seconds/image
```

</details>

![DiT_FasterDiffusion samples](visuals/infer_dit_fasterdiffusion.png)

## Evaluation
DiT provides a script for evaluation [sample_ddp.py](https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py).
The evaluation code is modified from this code, and the 50k sampling results are saved in the same data format as [ADM](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py).
The evaluation code is obtained from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations), and the evaluation environment is already included in the DiT-FasterDiffusion environment.


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
# Inception Score: 275.85888671875
# FID: 2.2854618308181784
# sFID: 4.575237382297701
# Precision: 0.82562
# Recall: 0.5825

echo 'DiT (FasterDiffusion):'
export CFG_SCALE=1.5
MODEL_FLAGS="--model DiT-XL/2 --num-fid-samples 50000 --image-size 256 --only-DiT False"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py $MODEL_FLAGS --cfg-scale $CFG_SCALE --per-proc-batch-size $BATCH_SIZE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz
# Inception Score: 276.0558166503906
# FID: 2.311326059070666
# sFID: 4.552778228460056
# Precision: 0.82654
# Recall: 0.5764
```

- DiT-XL/2-G (cfg=1.25)

<details>
<summary>Output:</summary>

```bash
#!/bin/bash

export NCCL_P2P_DISABLE=1

NUM_GPUS=8
BATCH_SIZE=32

echo 'DiT:'
export CFG_SCALE=1.25
MODEL_FLAGS="--model DiT-XL/2 --per-proc-batch-size 64 --num-fid-samples 50000 --image-size 256 --only-DiT True"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py  $MODEL_FLAGS --cfg-scale $CFG_SCALE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz
# Inception Score: 200.91566467285156
# FID: 3.2465643497795327
# sFID: 5.309920796937604
# Precision: 0.76142
# Recall: 0.635


echo 'DiT (FasterDiffusion):'
export CFG_SCALE=1.25
MODEL_FLAGS="--model DiT-XL/2 --num-fid-samples 50000 --image-size 256 --only-DiT False"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py $MODEL_FLAGS --cfg-scale $CFG_SCALE --per-proc-batch-size $BATCH_SIZE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz
# Inception Score: 
```

</details>


- DiT-XL/2

<details>
<summary>Output:</summary>

```bash
#!/bin/bash

export NCCL_P2P_DISABLE=1

NUM_GPUS=8
BATCH_SIZE=32

echo 'DiT:'
export CFG_SCALE=1.0
MODEL_FLAGS="--model DiT-XL/2 --per-proc-batch-size 64 --num-fid-samples 50000 --image-size 256 --only-DiT True"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py  $MODEL_FLAGS --cfg-scale $CFG_SCALE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz
# Inception Score: 122.8410415649414
# FID: 9.540725948224122
# sFID: 6.877126836603679
# Precision: 0.66544
# Recall: 0.6763

echo 'DiT (FasterDiffusion):'
export CFG_SCALE=1.0
MODEL_FLAGS="--model DiT-XL/2 --num-fid-samples 50000 --image-size 256 --only-DiT False"
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS sample_ddp_fasterdiffusion.py $MODEL_FLAGS --cfg-scale $CFG_SCALE --per-proc-batch-size $BATCH_SIZE
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples/DiT-XL-2-samples-50000.npz
# Inception Score: 
```

</details>


## BibTeX

```bibtex
@inproceedings{li2023faster,
  title={Faster diffusion: Rethinking the role of the encoder for diffusion model inference},
  author={Li, Senmao and van de Weijer, Joost and Khan, Fahad and Liu, Tao and Li, Linxuan and Yang, Shiqi and Wang, Yaxing and Cheng, Ming-Ming and others},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2023}
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