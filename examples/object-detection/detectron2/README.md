# Object Detection with Detectron2

This is an example of using [Detectron2](https://github.com/facebookresearch/detectron2) to train a model on the Zenseact Open Dataset.

- install detectron2 following the [official instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

- install the zod package

    `pip install zod`

- generate a dataset in COCO format, using the zod CLI:

    `zod generate coco --zod-dir /path/to/zod --output-dir /path/to/output`

- train a model using the Detectron2 CLI:

    `python train.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS /path/to/model.pth`

- evaluate the model using the detectron CLI:

    `python tools/eval_net.py --dataset coco_2017_val --cfg configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --load_ckpt /path/to/model.pth`

- visualize the model using the detectron CLI:

    `python tools/demo.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --input /path/to/image.jpg --output /path/to/output.jpg --opts MODEL.WEIGHTS /path/to/model.pth`

