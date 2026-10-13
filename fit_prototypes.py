import torch, argparse, os
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from gaussian_prototypes import GaussianPrototypes


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    # 全局原型：object_class='all'，一次拟合所有类的正常样本
    if args.dataset == 'mvtec':
        ds = MVTECDataset('train', object_class='all', rootdir=args.data_dir, transform=transform,
                          image_size=args.image_size, center_size=args.center_size,
                          augment=False, center_crop=args.center_crop)
    else:
        ds = VISADataset('train', object_class='all', rootdir=args.data_dir, transform=transform,
                         image_size=args.image_size, center_size=args.center_size,
                         augment=False, center_crop=args.center_crop)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_latents = []
    for x, _, _ in loader:
        with torch.no_grad():
            z = vae.encode(x.to(device)).latent_dist.mean.mul_(0.18215)   # [B, C, H, W]
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])                  # [B*H*W, C]
        all_latents.append(z.cpu())
    all_latents = torch.cat(all_latents, dim=0)   # [N, D]
    print('total latent tokens:', all_latents.shape)

    # 规模控制：随机子采样
    if all_latents.shape[0] > args.max_tokens:
        idx = torch.randperm(all_latents.shape[0])[:args.max_tokens]
        all_latents = all_latents[idx]

    proto = GaussianPrototypes(n_components=args.n_components)
    proto.fit(all_latents)

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f'{args.dataset}_k{args.n_components}.pt')
    proto.save(path)
    print('saved ->', path)
    print('mu', proto.mu.shape, 'logvar', proto.logvar.shape, 'pi', proto.pi)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='mvtec', choices=['mvtec', 'visa'])
    p.add_argument('--data-dir', default='./mvtec-dataset/')
    p.add_argument('--vae-type', default='ema')
    p.add_argument('--image-size', type=int, default=288)
    p.add_argument('--center-size', type=int, default=256)
    p.add_argument('--center-crop', type=lambda v: v.lower() in ('yes', 'true', 't', 'y', '1'), default=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--n-components', type=int, default=30)     # k
    p.add_argument('--max-tokens', type=int, default=300000)   # 子采样上限
    p.add_argument('--save-dir', default='./prototypes')
    args = p.parse_args()
    main(args)
