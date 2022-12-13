# Zenseact Open Dataset

TODO:  Move over stuff from the old README

## Installation

To install the full devkit, including the CLI, run:
```bash
pip install zod[cli]
```

Else, to install the library only, run:
```bash
pip install zod
```

## Download

nice little trick to download only mini version from cluster

```bash
rsync -ar --info=progress2 hal:/staging/dataset_donation/round_2/mini_train_val_single_frames.json ~/data/zod

cat ~/data/zod/mini_train_val_single_frames.json | jq -r '.[] | .[] | .frame_id' | xargs -I{} rsync -ar --info=progress2 hal:/staging/dataset_donation/round_2/single_frames/{} ~/data/zod/single_frames
```
