import imageio
import os
import glob
import math
import numpy as np
from skimage.metrics import structural_similarity

split = "val_view"

result_dir = f"/home/ruilongli/workspace/A-NeRF/render_output/zju_313_{split}/image/"
result_imgs = sorted(glob.glob(result_dir + "*.png"))

gt_dir = f"/home/ruilongli/workspace/implcarv/implcarv/data/collected/313/5/{split}/"
gt_imgs = sorted(glob.glob(gt_dir + "*.png"))

def compute_ssim(pred, target):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    try:
        ssim = structural_similarity(
            pred, target, channel_axis=-1
        )
    except ValueError:
        ssim = structural_similarity(
            pred, target, multichannel=True
        )
    return ssim

psnrs, ssims = [], []
for i, (result_img, gt_img) in enumerate(zip(result_imgs, gt_imgs)):
    result_img = imageio.imread(result_img)[:, :512] / 255.
    gt_img = imageio.imread(gt_img)[:, 512:] / 255.
    
    mse = ((result_img - gt_img) ** 2).mean()
    psnr = -10.0 * math.log(mse) / math.log(10.0)
    ssim = compute_ssim(result_img, gt_img)
    psnrs.append(psnr)
    ssims.append(ssim)
    print (psnr, ssim)

    canvas = np.hstack([result_img, gt_img]) * 255
    imageio.imwrite(result_dir + "%05d.png" % i, np.uint8(canvas))

psnr_avg = sum(psnrs) / len(psnrs)
ssim_avg = sum(ssims) / len(ssims)

print ("split:", split)
print ("avg psnr", psnr_avg)
print ("avg ssim", ssim_avg)