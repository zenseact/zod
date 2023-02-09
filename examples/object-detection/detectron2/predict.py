import argparse
import os.path as osp
import random

import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from train import build_config, register_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zod-path", default="data/zod/single_frames")
    parser.add_argument("--train-json", default="coco-style_full_train.json")
    parser.add_argument("--val-json", default="coco-style_full_val.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir", default="output/vis")
    args = parser.parse_args()

    register_dataset(args.zod_path, args.train_json, args.val_json)
    dataset = DatasetCatalog.get("zod/train")
    metadata = MetadataCatalog.get("zod/train")

    cfg = build_config(argparse.Namespace())
    cfg.MODEL.WEIGHTS = args.checkpoint
    predictor = DefaultPredictor(cfg)

    for d in random.sample(dataset, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.savefig(osp.join(args.save_dir, osp.basename(d["file_name"])))


if __name__ == "__main__":
    main()
