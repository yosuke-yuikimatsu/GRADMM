# GRADMM

This repository is the official implementation of our ICML 2025 paper [Synthetic Text Generation for Training Large Language Models via Gradient Matching](https://arxiv.org/pdf/2502.17607).

## 🔗 Quick Links
- [GRADMM](#gradmm)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Generation](#data-generation)
  - [Finetuning](#finetuning)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Install Requirements
```bash
conda create -n gradmm python=3.11
conda activate gradmm
pip install -r requirements.txt
```

## Data Generation
```bash
cd gradmm
./scripts/admm.sh
./scripts/admm_dp.sh
```

For filtering, please refer to the notebook `gradmm/Filtering.ipynb`. Adjust the settings in the `Parameters` section, then run all cells in the notebook.

### Distribution Matching Extension

This repository also includes an optional experimental Dataset Condensation with Distribution Matching (DM) objective. DM matches real and synthetic feature distributions after passing LM input embeddings through frozen random projectors. It can be used as a regularizer on top of the existing GRADMM/ADMM gradient-matching objective, or as a standalone objective with `--dm_mode standalone`.

The default GRADMM behavior is unchanged unless `--use_dm True` is set. A basic DM-regularized generation run is available via:

```bash
cd gradmm
./scripts/dm.sh
```

You can also add DM flags to an existing generation command, for example:

```bash
python generate.py \
  --dataset sst2 \
  --split validation \
  --use_dm True \
  --dm_mode regularizer \
  --dm_weight 1.0 \
  --dm_projector mlp \
  --dm_match mean
```

For a randomly initialized BERT feature space, use a smaller weight to limit overhead:

```bash
python generate.py \
  --dataset sst2 \
  --split validation \
  --use_dm True \
  --dm_projector random_bert \
  --dm_weight 0.1
```

### Stable DM smoke tests

These checks are ordered from cheapest to most numerically demanding. The stabilizing flags are optional and are mainly intended for debugging: `--force_float32_model true` loads the language model in fp32 and may require more VRAM, while `--embed_value_clip` clamps ADMM embedding variables after updates.

#### A. CPU unit tests

```bash
cd gradmm
python test_dm_smoke.py
```

#### B. Adam DM smoke test (recommended first)

```bash
cd gradmm
WANDB_MODE=disabled WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python generate.py \
  --rng_seed 42 \
  --dataset sst2 \
  --split validation \
  --batch_size 4 \
  --n_steps 5 \
  --n_gen_samples 4 \
  --subset_size 4 \
  --n_gen 2 \
  --gen_bs 2 \
  --init_candidates 5 \
  --use_auto_gen_tokens true \
  --print_full false \
  --print_every 1 \
  --save_every 1 \
  --model_name phi \
  --opt_alg adam \
  --loss dlg \
  --embed_loss dlg \
  --lr 0.00001 \
  --work_base_dir ./synthetic_data/smoke_dm_adam_5steps \
  --overwrite true \
  --grad_clip 0.05 \
  --gen_grad_clip "" \
  --topk 20 \
  --use_dm true \
  --dm_mode regularizer \
  --dm_weight 0.01 \
  --dm_projector mlp \
  --dm_feature_dim 32 \
  --dm_match mean \
  --dm_num_projectors 1 \
  --dm_by_class true \
  --dm_real_batch_size 4
```

#### C. Conservative ADMM smoke test (after Adam works)

```bash
cd gradmm
WANDB_MODE=disabled WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python generate.py \
  --rng_seed 42 \
  --dataset sst2 \
  --split validation \
  --batch_size 4 \
  --n_steps 1 \
  --n_gen_samples 4 \
  --subset_size 4 \
  --n_gen 2 \
  --gen_bs 2 \
  --init_candidates 5 \
  --use_auto_gen_tokens true \
  --print_full false \
  --print_every 1 \
  --save_every 1 \
  --model_name phi \
  --opt_alg admm \
  --loss dlg \
  --embed_loss dlg \
  --admm_rho 1.0 \
  --admm_inner_steps 1 \
  --lr 0.00001 \
  --work_base_dir ./synthetic_data/smoke_dm_admm_safe \
  --overwrite true \
  --grad_clip 0.05 \
  --gen_grad_clip "" \
  --topk 20 \
  --drop_non_english_tokens true \
  --use_dm true \
  --dm_mode regularizer \
  --dm_weight 0.01 \
  --dm_projector mlp \
  --dm_feature_dim 32 \
  --dm_match mean \
  --dm_num_projectors 1 \
  --dm_by_class true \
  --dm_real_batch_size 4 \
  --embed_value_clip 5.0
```

#### Regression smoke checks

* Re-run either command with the same `--work_base_dir` and `--overwrite true`; fresh run state should be initialized each time.
* Add `--print_full true --conversion_method topk --use_dm true` to the Adam command to verify top-k full printing does not raise `loss_dict` initialization errors.
* For difficult ADMM runs, prefer `--drop_non_english_tokens true`, consider `--embed_value_clip 5.0`, and use `--force_float32_model true` if fp16/bf16 instability is suspected.

## Finetuning
1. Obtain the synthetic data paths by running the `Print fine-tuning paths` section in the notebook `gradmm/Finetuning.ipynb`.

2. Insert the retrieved paths into `scripts/query_ft.sh`, then run the following commands:
```bash
cd addax
./scripts/query_ft.sh
```

3. To collect the fine-tuning results, paste the fine-tuning paths into `Collect fine-tuning results` section in the notebook `gradmm/Finetuning.ipynb` and un the corresponding cells.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Dang Nguyen (nguyentuanhaidang@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{nguyen2025synthetic,
  title={Synthetic Text Generation for Training Large Language Models via Gradient Matching},
  author={Nguyen*, Dang and Li*, Zeman and Bateni, Mohammadhossein and Mirrokni, Vahab and Razaviyayn, Meisam and Mirzasoleiman, Baharan},
  journal={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Acknowledgements
The structure of this repository is largely based on the official implementation of [lamp](https://github.com/eth-sri/lamp) and [Addax](https://github.com/optimization-for-data-driven-science/Addax). We are grateful for their open sources.