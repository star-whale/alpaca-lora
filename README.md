# 🦙🌲🤏 Alpaca-LoRA && Starwhale

This repo is forked from [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora), see the [original README.md](README_original.md). We add some Starwhale support for this repo, users could manage lifecycle of the model/dataset by starwhale, including:
- finetune a new version of model locally or remotelly and get a finetuned version of model
- serve an API locally or remotelly
- evaluate the model with Starwhale datasets

## Build Starwhale Datasets

```python
python build_swds.py
```

## Build Starwhale Model

```python
python build_swmp.py
```

## Finetune the model with dataset and gain a new verson

```bash
swcli model run -u llama-7b-hf/version/latest -d test/version/latest -h swmp_handlers:fine_tune
```

## Serve an API for the finetuned version of model

```bash
swcli model serve -u llama-7b-hf/version/latest
```
