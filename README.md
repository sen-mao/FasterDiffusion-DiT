## Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-FasterDiffusion-<COLOR>.svg)](https://arxiv.org/abs/2312.09608) [![arXiv](https://img.shields.io/badge/paper-FasterDiffusion-b31b1b.svg)](https://arxiv.org/abs/2312.09608.pdf)
> **Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models**
>
> [Senmao Li](https://github.com/sen-mao), [Taihang Hu](https://github.com/hutaiHang), [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en&oi=sra), [Fahad Khan](https://sites.google.com/view/fahadkhans/home), [Tao Liu](ltolcy0@gmail.com), [Linxuan Li](https://github.com/Potato-lover), [Shiqi Yang](https://www.shiqiyang.xyz/), [Yaxing Wang](https://yaxingwang.netlify.app/author/yaxing-wang/), [Ming-Ming Cheng](https://mmcheng.net/), [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en)

The official codebase for [FasterDiffusion](https://arxiv.org/abs/2312.09608) accelerates [DiT](https://github.com/facebookresearch/DiT) with **~1.51x** speedup.


![DiT samples](visuals/infer_dit.jpg)

Results of DiT (top) and this method in conjunction with our proposed approach (bottom).

## Requirements

First, download and set up the repo:

```bash
git clone https://github.com/sen-mao/DiT-FasterDiffusion.git
cd DiT
```

A suitable conda environment named `DiT-FasterDiffusion` can be created
and activated with:

```
conda env create -f environment.yaml

conda activate DiT-FasterDiffusion
```

## Performance

| DiT Model                                                                                 | Resolution | FID&darr; | sFID&darr; | IS&uarr; | Precision&uarr; | Recall&uarr; | s/image&darr; |
|-------------------------------------------------------------------------------------------|:----------:|:---------:|:----------:|:--------:|:---------------:|:------------:|:-------------:|
| [XL/2-G](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)                   |  256x256   |   2.27    |    4.60    |  278.24  |      0.83       |     0.57     |     5.13      |
| [XL/2-G](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) (FasterDiffusion) |  256x256   |   2.31    |    4.55    |  276.05  |      0.82       |     0.57     |     3.26      |


# DiT-XL/2-G

### Run the `sample_fasterdiffusion.py` to generate images.
```bash
python sample_fasterdiffusion.py --only-DiT 1 
```


```bash
key time-steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
Warming up GPU...
100%|█████████████████████████████████████████| 250/250 [00:41<00:00,  6.09it/s]
100%|█████████████████████████████████████████| 250/250 [00:40<00:00,  6.16it/s]
DiT: 5.11 seconds/image
```

### Run the `sample_fasterdiffusion.py` to generate images with FasterDiffusion.
```bash
python sample_fasterdiffusion.py 
```

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
