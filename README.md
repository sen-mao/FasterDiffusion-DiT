## Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models

<a href='https://arxiv.org/abs/2312.09608'><img src='https://img.shields.io/badge/ArXiv-2306.05414-red'></a>

> **Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models**
>
> [Senmao Li](https://github.com/sen-mao)\*, [Taihang Hu](https://github.com/hutaiHang), [Fahad Khan](https://sites.google.com/view/fahadkhans/home), [Linxuan Li](https://github.com/Potato-lover), [Shiqi Yang](https://www.shiqiyang.xyz/), [Yaxing Wang](https://yaxingwang.netlify.app/author/yaxing-wang/), [Ming-Ming Cheng](https://mmcheng.net/), [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en)
>
> ***Indicates that the author implemented the code.**

The official codebase for [FasterDiffusion](https://arxiv.org/abs/2312.09608) accelerates [DiT](https://github.com/facebookresearch/DiT) with **~1.51x** speedup.


![DiT samples](visuals/infer_dit.jpg)

Results of DiT (top) and this method in conjunction with our proposed approach (bottom).

## Requirements

First, download and set up the repo:

```bash
git clone https://github.com/sen-mao/DiT.git
cd DiT
```

A suitable conda environment named `DiT-faster-diffusion` can be created
and activated with:

```
conda env create -f environment.yaml

conda activate DiT-faster-diffusion
```

## Performance

| Model                  | Dataset | Resolution | FID&darr; | sFID&darr; | IS&uarr; | Precision&uarr; | Recall&uarr; | s/image&darr; |
|------------------------|:--------:|:----------:|:---------:|:----------:|:--------:|:---------------:|:------------:|:-------------:|
| DiT                    | ImageNet |  256x256   |   2.27    |    4.60    |  278.24  |      0.83       |     0.57     |     5.13      |
| DiT w/ FasterDiffusion | ImageNet |  256x256   |   2.31    |    4.55    |  276.05  |      0.82       |     0.57     |     **3.26**      |
| DiT                    | ImageNet |  512x512   |   3.04    |    5.02    |  240.82  |      0.84       |     0.54     |     26.25     |
| DiT w/ FasterDiffusion | ImageNet |  521x512   |   3.25    |    5.05    |  245.13  |      0.83       |     0.51     |     **17.35**     |


# Visualization

Run the `infer_dit.py` to generate images with FasterDiffusion.


## BibTeX

```bibtex
@article{li2023faster,
  title={Faster diffusion: Rethinking the role of unet encoder in diffusion models},
  author={Li, Senmao and Hu, Taihang and Khan, Fahad Shahbaz and Li, Linxuan and Yang, Shiqi and Wang, Yaxing and Cheng, Ming-Ming and Yang, Jian},
  journal={arXiv preprint arXiv:2312.09608},
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
