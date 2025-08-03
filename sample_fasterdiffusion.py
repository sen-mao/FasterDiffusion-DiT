# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import time

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/data/stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    ## For fasterdiffusion
    diffusion.register_store = {'se_step': False, 'mid_feature': None,
                                'key_time_steps': list(range(args.num_sampling_steps+1)),
                                'init_img': None, 'use_parallel': True, 'ts_parallel': None, 'steps': [0],
                                'bs': len(class_labels), 'tqdm_disable': False, 'noise_injection': True}

    diffusion.register_store['key_time_steps'] = \
        [0, 1, 2, 3, 4, 5, 6, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22, 25, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 45, 49,
         51, 52, 53, 54, 58, 59, 60, 62, 64, 67, 68, 70, 72, 73, 75, 77, 78, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92,
         95, 97, 99, 101, 102, 103, 104, 106, 107, 109, 110, 111, 112, 115, 116, 117, 118, 119, 120, 121, 123, 125, 128,
         130, 131, 132, 134, 137, 138, 139, 141, 142, 143, 144, 145, 147, 150, 152, 155, 157, 158, 160, 161, 162, 163,
         164, 166, 168, 169, 170, 172, 173, 174, 175, 178, 180, 181, 182, 184, 187, 188, 189, 190, 191, 193, 194, 195,
         196, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 214, 216, 217, 218, 219, 221, 223,
         224, 225, 226, 227, 228, 229, 231, 232, 233, 234, 237, 239, 241, 242, 243, 245, 246, 247, 249, 250]

    # DiT w/o fasterdiffusion
    if args.only_DiT:
        diffusion.register_store['key_time_steps'] = list(range(args.num_sampling_steps+1))

    print('key time-steps =', diffusion.register_store['key_time_steps'])
    print(len(diffusion.register_store['key_time_steps']))

    # Create sampling noise:
    n = len(class_labels)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Warmup GPU. Only for testing the speed.
    print("Warming up GPU...")
    for _ in range(2):
        _ = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
            progress=not diffusion.register_store['tqdm_disable'], device=device
        )

    # Sample images:
    diffusion.register_store['tqdm_disable'] = True  # if one wants to disable `tqdm`
    start_time = time.time()
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=not diffusion.register_store['tqdm_disable'], device=device
    )
    use_time = time.time() - start_time
    print(f"DiT {'' if args.only_DiT else '(FasterDiffusion)'}: {use_time / len(class_labels):.2f} seconds/image")

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--only-DiT", type=bool, default=False)
    args = parser.parse_args()
    main(args)
