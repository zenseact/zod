# Zenseact Open Dataset

[![Stable Version](https://img.shields.io/pypi/v/zod?label=stable)](https://pypi.org/project/zod/#history)
[![Python Versions](https://img.shields.io/pypi/pyversions/zod)](https://pypi.org/project/zod/)
[![Download Stats](https://img.shields.io/pypi/dm/zod)](https://pypistats.org/packages/zod)

The Zenseact Open Dataset (ZOD) is a large multi-modal autonomous driving dataset developed by a team of researchers at [Zenseact](https://zenseact.com/). The dataset is split into three categories: *Frames*, *Sequences*, and *Drives*. For more information about the dataset, please refer to our [paper](https://arxiv.org/abs/2305.02008), or visit our [website](https://zod.zenseact.com).

## Examples
Find examples of how to use the dataset in the [examples](examples/) folder. Here you will find a set of jupyter notebooks that demonstrate how to use the dataset, as well as an example of how to train an object detection model using [Detectron2](https://github.com/facebookresearch/detectron2).

## Installation

The install the library with minimal dependencies, for instance, to be used in a training environment without the need for interactivity or visualization, run:
```bash
pip install zod
```

To install the library along with the CLI, which can be used to download the dataset, convert between formats, and perform visualization, run:
```bash
pip install "zod[cli]"
```

To install the full devkit, with the CLI and all dependencies, run:
```bash
pip install "zod[all]"
```

## Download using the CLI

This is an example of how to download the ZOD Frames mini-dataset using the CLI. Prerequisites are that you have applied for access and received a download link.
The simplest way to download the dataset is to use the CLI interactively:
```bash
zod download
```
This will prompt you for the required information, present you with a summary of the download, and then ask for confirmation. You can of course also specify all the required information directly on the command line, and avoid the confirmation using `--no-confirm` or `-y`. For example:
```bash
zod download -y --url="<download-link>" --output-dir=<path/to/outputdir> --subset=frames --version=mini
```
By default, all data streams are downloaded for ZodSequences and ZodDrives. For ZodFrames, DNAT versions of the images, and surrounding (non-keyframe) lidar scans are excluded. To download them as well, run:
```bash
zod download -y --url="<download-link>" --output-dir=<path/to/outputdir> --subset=frames --version=full --num-scans-before=-1 --num-scans-after=-1 --dnat
```
If you want to exclude some of the data streams, you can do so by specifying the `--no-<stream>` flag. For example, to download only the DNAT images, infos, and annotations, run:
```bash
zod download --dnat --no-blur --no-lidar --no-oxts --no-vehicle-data
```
Finally, for a full list of options you can of course run:
```bash
zod download --help
```

## Anonymization
To preserve privacy, the dataset is anonymized. The anonymization is performed by [brighterAI](https://brighter.ai/), and we provide two separate modes of anonymization: deep fakes (DNAT) and blur. In our paper, we show that the performance of an object detector is not affected by the anonymization method. For more details regarding this experiment, please refer to our [coming soon]().

## Citation
If you publish work that uses Zenseact Open Dataset, please cite [our paper](https://arxiv.org/abs/2305.02008):

```
@inproceedings{zod2023,
  author = {Alibeigi, Mina and Ljungbergh, William and Tonderski, Adam and Hess, Georg and Lilja, Adam and Lindstr{\"o}m, Carl and Motorniuk, Daria and Fu, Junsheng and Widahl, Jenny and Petersson, Christoffer},
  title = {Zenseact Open Dataset: A large-scale and diverse multimodal dataset for autonomous driving},
  year = {2023},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20178--20188},
}
```

## Contact
For questions about the dataset, please [Contact Us](mailto:opendataset@zenseact.com).

## Contributing
We welcome contributions to the development kit. If you would like to contribute, please open a pull request.

## License
**Dataset**:
This dataset is the property of Zenseact AB (© 2023 Zenseact AB) and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Any public use, distribution, or display of this dataset must contain this notice in full:

> For this dataset, Zenseact AB has taken all reasonable measures to remove all personally identifiable information, including faces and license plates. To the extent that you like to request the removal of specific images from the dataset, please contact [privacy@zenseact.com](mailto:privacy@zenseact.com).

The purpose of Zenseact AB is to save lives in road traffic. We encourage use of this dataset with the intention of avoiding losses in road traffic. ZOD is not intended for military use.

**Development kit**:
This development kit is the property of Zenseact AB (© 2023 Zenseact AB) and is licensed under [MIT](https://opensource.org/licenses/MIT).
