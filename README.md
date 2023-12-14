# Voice Anti Spoofing

## Report

You may see the report if you follow [this link](https://api.wandb.ai/links/l0u1za/7h6vtv2b)

## Installation guide

Firstly, install needed requirements for running model

```shell
pip install -r ./requirements.txt
```

### Download model

Use bash script to download trained model

```shell
cd ./default_test_model
./download.sh
```

It will be placed to `./default_test_model/checkpoint.pth`

If you have some issues using bash utilities, you may download model directly from [google drive](https://drive.google.com/file/d/1pdkOtX1fvfQucxvaFLkWi7uxR6fq7xSW/view?usp=sharing)


## Run test model with prepared configuration

You may check this model on your data. Put your audio to test folder `test_data/audio`.
```shell
python test.py \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```

Or you can check EER and predictions on ASVDataset. Then provide config link to arguments
```shell
python test.py \
   -r default_test_model/checkpoint.pth \
   -c default_test_config.json \
   -o test_result.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash
docker build -t my_src_image .
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_src_image python -m unittest
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
