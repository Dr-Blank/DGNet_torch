import os
import torch
import argparse
import numpy as np
from scipy import misc
import imageio
from pathlib import Path

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

from utils.dataset import test_dataset as EvalDataset
from lib import DGNet, ModelParams


def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(
        image_root=val_root + "Imgs/", gt_root=val_root + "GT/", testsize=trainsize
    )

    model.eval()
    with torch.no_grad():
        for _ in range(val_loader.size):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            output = model(image)
            output = F.upsample(
                output[0], size=gt.shape, mode="bilinear", align_corners=False
            )
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            imageio.imwrite(map_save_path + name, output)
            print(">>> saving prediction at: {}".format(map_save_path + name))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="DGNet",
        choices=[
            "DGNet",
            "DGNet-S",
            "DGNet-PVTv2-B0",
            "DGNet-PVTv2-B1",
            "DGNet-PVTv2-B2",
            "DGNet-PVTv2-B3",
            "DGNet-PVTv2-B4",
        ],
    )
    parser.add_argument(
        "--snap_path",
        type=str,
        default="./snapshot/Exp-DGNet/Net_epoch_best.pth",
        help="snapshot path of trained model",
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="train use gpu")
    opt = parser.parse_args()

    txt_save_path = Path("./result/{}/".format(opt.snap_path.split("/")[-2]))
    txt_save_path.mkdir(parents=True, exist_ok=True)

    print(">>> configs:", opt)

    # set the device for training
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(f"USE GPU {opt.gpu_id}")

    cudnn.benchmark = True

    model = DGNet(**ModelParams.get_params(opt.model).__dict__).cuda()

    model.load_state_dict(torch.load(opt.snap_path))
    model.eval()

    # for data_name in ['CamouflageData']:
    # # for data_name in ['CAMO', 'COD10K', 'NC4K']:
    #     map_save_path = str(txt_save_path) + "{}/".format(data_name)
    #     os.makedirs(map_save_path, exist_ok=True)
    #     evaluator(
    #         model=model,
    #         val_root='./dataset/TestDataset/' + data_name + '/',
    #         map_save_path=map_save_path,
    #         trainsize=352)

    MOCA = Path("./dataset/TestDataset/MOCA")

    # for every folder in MOCA
    for folder in MOCA.iterdir():
        map_save_path = txt_save_path / MOCA.name / folder.name
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root=str((MOCA / folder.name).resolve()) + "/",
            map_save_path=str(map_save_path.resolve()) + "/",
            trainsize=352,
        )


if __name__ == "__main__":
    main()
