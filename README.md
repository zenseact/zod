# Zenseact Open Dataset
The Zenseact Open Dataset (ZOD) is a large multi-modal autonomous driving dataset developed by a team of researchers at [Zenseact](https://zenseact.com/). The dataset is split into three categories: *Frames*, *Sequences*, and *Drives*. For more information about the dataset, please refer to our [coming soon](), or visit our [website](https://zenseact.github.io/zod-web/).

## Installation

To install the full devkit, including the CLI, run:
```bash
pip install zod[cli]
```

Else, to install the library only, run:
```bash
pip install zod
```

## Download (to-be-deleted before release)

nice little trick to download only mini version from cluster

```bash
mkdir ~/data/zod
rsync -ar --info=progress2 hal:/staging/dataset_donation/round_2/mini_train_val_single_frames.json ~/data/zod/

cat ~/data/zod/mini_train_val_single_frames.json | jq -r '.[] | .[] | .id' | xargs -I{} rsync -ar --info=progress2 hal:/staging/dataset_donation/round_2/single_frames/{} ~/data/zod/single_frames
```

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

## License
The Zenseact Open Dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and the development-kit (this repository) is licensed under [MIT](https://opensource.org/licenses/MIT)