# Zenseact Open Dataset
The Zenseact Open Dataset (ZOD) is a large multi-modal autonomous driving dataset developed by a team of researchers at [Zenseact](https://zenseact.com/). The dataset is split into three categories: *Frames*, *Sequences*, and *Drives*. For more information about the dataset, please refer to our [coming soon](), or visit our [website](https://zenseact.github.io/zod-web/).

## Examples
Find examples of how to use the dataset in the [examples](examples/) folder. Here you will find a set of jupyter notebooks that demonstrate how to use the dataset, as well as an example of how to train an object detection model using [Detectron2](https://github.com/facebookresearch/detectron2).

## Installation

To install the full devkit, including the CLI, run:
```bash
pip install zod[cli]
```

Else, to install the library only, run:
```bash
pip install zod
```

## Download using the CLI

This is an example of how to download the ZOD Frames mini-dataset using the CLI. Prerequisites are that you have applied for access and received a download link. To download the mini-dataset, run:
```bash
zod download --url <download-link> --output-dir <path/to/outputdir> frames --mini
```
similarly, to download the full dataset, run:
```bash
zod download --url <download-link> --output-dir <path/to/outputdir> frames --lidar -num-scans-before -1 --num-scans-after -1 --oxts --images --blur --dnat ---calibrations --annotations
```
this will download all the previous and future lidar scans (num-scans-before and num-scans-after), the OxTS data, the images (with both the blur and DNAT anonymization), the calibration files, the annotations, and all other necessary files. For a full list of options for ZOD download, run:
```bash
zod download frames --help
zod download sequences --help
```
depending on which dataset you want to download.


## Anonymization
To preserve privacy, the dataset is anonymized. The anonymization is performed by [brighterAI](https://brighter.ai/), and we provide two separate modes of anonymization: deep fakes (DNAT) and blur. In our paper, we show that the performance of an object detector is not affected by the anonymization method. For more details regarding this experiment, please refer to our [coming soon]().

## Citation
If you publish work that uses Zenseact Open Dataset, please cite: [coming soon]()

```
@misc{zod2021,
  author = {TODO},
  title = {Zenseact Open Dataset},
  year = {2023},
  publisher = {TODO},
  journal = {TODO},
```

## Contact
For questions about the dataset, please [Contact Us](mailto:opendataset@zenseact.com).

## Contributing
We welcome contributions to the development kit. If you would like to contribute, please open a pull request.

## License
**Dataset**:
This dataset is the property of Zenseact AB (© 2023 Zenseact AB) and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Any public use, distribution, or display of this dataset must contain this notice in full:

> For this dataset, Zenseact AB has taken all reasonable measures to remove all personally identifiable information, including faces and license plates. To the extent that you like to request the removal of specific images from the dataset, please contact [privacy@zenseact.com](mailto:privacy@zenseact.com).


**Development kit**:
This development kit is the property of Zenseact AB (© 2023 Zenseact AB) and is licensed under [MIT](https://opensource.org/licenses/MIT).
