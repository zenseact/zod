import os
import os.path as osp
from argparse import Namespace

from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from zod.constants import ALL_CLASSES


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, cfg.RESULT_SUBDIR)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def build_config(args: Namespace) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("zod/train",)
    cfg.DATASETS.TEST = ("zod/val",)
    cfg.DATALOADER.NUM_WORKERS = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.RESULT_SUBDIR = args.result_subdir
    if not args.use_imagenet:
        # Disable ImageNet pre-training.
        # We must lower LR and increase warmup avoid diverging in the beginning
        cfg.MODEL.WEIGHTS = None
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.WARMUP_ITERS = 50000
        cfg.SOLVER.WARMUP_FACTOR = 0.1
        cfg.SOLVER.WARMUP_METHOD = "constant"
        cfg.SOLVER.STEPS = [step + cfg.SOLVER.WARMUP_ITERS for step in cfg.SOLVER.STEPS]
        cfg.SOLVER.MAX_ITER += cfg.SOLVER.WARMUP_ITERS
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def register_dataset(zod_path, train_json, val_json):
    register_coco_instances("zod/train", {}, osp.join(zod_path, "coco-style", train_json), zod_path)
    register_coco_instances("zod/val", {}, osp.join(zod_path, "coco-style", val_json), zod_path)


def main(args: Namespace):
    register_dataset(args.zod_path, args.train_json, args.val_json)
    cfg = build_config(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.eval_only:
        trainer.test(cfg, trainer.model)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--zod-path", default="data/zod/single_frames")
    parser.add_argument("--train-json", default="zod_full_blur_png_train.json")
    parser.add_argument("--val-json", default="zod_full_blur_png_val.json")
    parser.add_argument("--output-dir", default="output/train")
    parser.add_argument("--result-subdir", default="results")
    parser.add_argument("--use-imagenet", action="store_true")
    args = parser.parse_args()
    if args.eval_only and not args.resume:
        raise ValueError("Must resume from a checkpoint when evaluating")
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
