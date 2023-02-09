import argparse
import random

import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from .train import register_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zod-path", default="data/zod/single_frames")
    parser.add_argument("--train-json", default="zod_full_blur_png_train.json")
    parser.add_argument("--val-json", default="zod_full_blur_png_val.json")
    args = parser.parse_args()

    register_dataset(args.zod_path, args.train_json, args.val_json)
    dataset = DatasetCatalog.get("zod/train")
    metadata = MetadataCatalog.get("zod/train")

    for d in random.sample(dataset, 2):
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=4)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(38, 18), dpi=50, frameon=False)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
