# Vocoder project

## Report

You may see the report if you follow [this link](https://wandb.ai/l0u1za/nv_project/reports/Vocoder--Vmlldzo2MTUwNDMz?accessToken=a3uw6fk71k1ind39vg6diiiss63qswtd4f9h8u6m9sigusc91wxa5qqjkczqqlsg)

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

If you have some issues using bash utilities, you may download model directly from [google drive](https://drive.google.com/file/d/1ZAu0x9M6A_xAr-h_fCLjbZqSlB9WftNe/view?usp=sharing)


## Run test model with prepared configuration

```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
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
