import argparse
import os
import torch
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image

from test import test_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description='Kanji stylization: transfer the style of a SINGLE fixed '
                    'handwritten reference onto every synthetic kanji glyph '
                    '(1-to-1, no cartesian product).')
    # Basic options
    parser.add_argument('--content_dir', type=str, default='data/kanji_content',
                        help='Directory path to the batch of synthetic kanji '
                             '(content) images')
    parser.add_argument('--style_dir', type=str, default='data/kanji_probe_style',
                        help='Directory path to the handwritten style images; '
                             'a single one is picked deterministically '
                             'as the style reference')
    parser.add_argument('--decoder', type=str, default='experiments/ArtFlow-Kanji/glow.pth',
                        help='path for the fine-tuned decoder model')
    # Additional options
    parser.add_argument('--size', type=int, default=256,
                        help='New size for the content and style images')
    parser.add_argument('--style_index', type=int, default=0,
                        help='Index (after sorting) of the style reference to '
                             'use for the whole batch')
    parser.add_argument('--save_ext', default='.png',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output_kanji',
                        help='Directory to save the output image(s)')
    # Glow net config (must match the fine-tuned checkpoint)
    parser.add_argument('--n_flow', default=8, type=int,
                        help='number of flows in each block')
    parser.add_argument('--n_block', default=2, type=int,
                        help='number of blocks')
    parser.add_argument('--affine', default=False, type=bool,
                        help='use affine coupling instead of additive')
    parser.add_argument('--no_lu', action='store_true',
                        help='disable LU decomposition in invertible 1x1 conv')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    from glow_adain import Glow

    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]
    content_paths.sort()
    print(f"{len(content_paths)} content images in {content_dir}")

    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
    style_paths.sort()
    if not style_paths:
        raise RuntimeError(f"no style references found in {style_dir}")
    style_path = style_paths[args.style_index % len(style_paths)]
    print(f"single style reference ({args.style_index}): {style_path}")

    # glow
    glow = Glow(3, args.n_flow, args.n_block, affine=args.affine,
                conv_lu=not args.no_lu)

    # -----------------------resume training------------------------
    glow = glow.to(device)
    if os.path.isfile(args.decoder):
        print("--------loading checkpoint----------")
        checkpoint = torch.load(args.decoder, weights_only=False)
        glow.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.decoder))
    else:
        raise RuntimeError("no checkpoint found: {}".format(args.decoder))
    glow = glow.to(device)
    for p in glow.parameters():
        p.requires_grad = False
    glow.eval()

    # -----------------------start------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # one style reference for the whole batch
    with torch.no_grad():
        style = Image.open(str(style_path)).convert('RGB')
        img_transform = test_transform(style, args.size)
        style = img_transform(style)
        style = style.to(device).unsqueeze(0)
        z_s = glow(style, forward=True)

        for content_path in content_paths:
            content = Image.open(str(content_path)).convert('RGB')
            img_transform = test_transform(content, args.size)
            content = img_transform(content)
            content = content.to(device).unsqueeze(0)

            # content/style ---> z ---> stylized
            z_c = glow(content, forward=True)
            output = glow(z_c, forward=False, style=z_s)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized{:s}'.format(
                content_path.stem, args.save_ext)
            print(output_name)
            save_image(output, str(output_name))

    print(f"done: {len(content_paths)} stylized kanji in {output_dir}")


if __name__ == '__main__':
    main()
