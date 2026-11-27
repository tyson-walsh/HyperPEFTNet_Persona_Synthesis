# HyperPEFTNet: Parameter-Efficient Hypernetworks for Persona Synthesis

Code, model weights and pseudonymous author-level data for the paper by Tyson Walsh, Trenton W. Ford and Daniel Runfola, Proceedings of the 32nd International Conference on Intelligent User Interfaces (IUI '27), Helsinki, Finland.

HyperPEFTNet gives every user of a language model their own persona without storing a model per user. Each user is described by six numbers measured from their writing and activity. A small network, the hypernetwork, turns those six numbers into parameter-efficient fine-tuning (PEFT) offsets for a frozen Pythia-160M decoder, so each user costs 24 bytes of storage and switching users needs no adapter file.

## What is in this repository

| folder | contents |
|---|---|
| `data/` | the six-number descriptor of each of the 10,000 users, and the three cohort label files the probes use |
| `weights/hyperpeft_lora/` | the trained HyperPEFT-LoRA generator, its input order, its training settings, and its learned reply-boundary rows |
| `weights/shared_heads/` | the four shared PEFT heads the paper compares against: LoRA, bottleneck adapter, bias-only and prefix |
| `weights/backbone_pythia_160m/` | the backbone's configuration and tokenizer; its weights file is attached to the GitHub release |
| `training/` | the training scripts for the generator and for the shared heads |
| `data_scripts/` | feature building, cohort labels, datasets and the hypernetwork model |
| `PEFT_scripts/` | the four PEFT modules |
| `evaluation/` | held-out likelihood, generation metrics and the hidden-state probes |
| `analysis/` | the paired likelihood tests, the probe statistics, the latency benchmark and the text-reuse check |
| `public_data/` | a converter from the released CSV files to the inputs the probe script reads |
| `MANIFEST.sha256` | the SHA-256 checksum of every file |

## The data

`data/descriptors.csv` has one row per user and seven columns: `user_key` and the six descriptor features, in the order the generator reads them (`weights/hyperpeft_lora/feature_names.json`): first-person ratio, punctuation ratio, question ratio, top-subreddit share, subreddit entropy and type-token ratio. Each feature is computed from the user's training-partition posts only, standardized over the 10,000 users, and rounded to two decimals. Because of the rounding, these values differ slightly from the full-precision inputs used in the paper's experiments.

Each `user_key` is a fresh random 12-character key. It cannot be derived from a user's identity, and the same keys are used in the three cohort files.

| file | cohort | labels |
|---|---|---|
| `data/cohort_question_rate.csv` | the top and bottom 20% of users by question rate | `high` 2,000, `low` 2,000 |
| `data/cohort_profanity.csv` | the top and bottom 15% by profanity rate | `high` 1,500, `low` 1,500 |
| `data/cohort_sentiment.csv` | the top and bottom 10% by sentiment polarity | `positive` 1,000, `negative` 1,000 |

The users were drawn from the public Pushshift Reddit archive for 2010 to 2016.

## The weights

- `weights/hyperpeft_lora/final_hypernet.safetensors`: the generator at the end of training (two passes over all 10,000 users). It writes LoRA offsets for the query, key and value projections of every layer.
- `weights/hyperpeft_lora/train_checkpoints/checkpoint_last.pt`: the reply-boundary rows HyperPEFT-LoRA learns for its two boundary tokens, in the token embedding and in the LM head (four rows, 3,072 values). The evaluation script loads them from this path.
- `weights/shared_heads/<head>/peft_placeholders.safetensors`: one shared head per PEFT family, trained on all users with the same data and schedule.
- `weights/backbone_pythia_160m/`: the frozen backbone is EleutherAI's Pythia-160M. Its weights file, `model.safetensors` (649,308,728 bytes, SHA-256 `b0942f4b31d83c5bc1e7f97785900e5c9566d11bd7901bce1b28f8eb5750c0c4`), is attached to release v1.0 of this repository: https://github.com/tyson-walsh/HyperPEFTNet_Persona_Synthesis/releases/download/v1.0/model.safetensors. Download it into this folder before running the scripts (the first command under "Running the code").

## Running the code

The scripts need Python 3.9 or later and the packages in `requirements.txt` (tested with Python 3.9.25). Each script prints its options with `--help`.

To run the hidden-state probes on the released data, first convert the CSV files into the inputs the probe script reads. The converter keeps the released values as they are (already standardized and rounded) and builds an integer index from the public keys alone:

```bash
curl -L -o weights/backbone_pythia_160m/model.safetensors https://github.com/tyson-walsh/HyperPEFTNet_Persona_Synthesis/releases/download/v1.0/model.safetensors
python public_data/prepare_public_inputs.py --data data --out public_inputs
python evaluation/validate_models_10000.py --help
```

`public_inputs/global_features.parquet` goes to the probe script's `--global_features` option and the three `public_inputs/labels_*.csv` files are its cohort labels. Because the released descriptors are rounded to two decimals, scores computed from them can differ slightly from the paper's, which used full-precision values.

The held-out-reply experiments need the conversation corpus itself, which is not redistributed:

1. Build each user's features and the cohort labels from your own copy of the corpus with `data_scripts/hypernetwork_feature_builder_10000.py` and `data_scripts/create_labels.py`.
2. Train a generator with `training/train_hypernet_160M.py --peft_head lora` (or `adapter`, `bias`, `prefix`) and the shared heads with `training/train_nonhypernet_10000.py`. Every generator reads the same six descriptor features, in the order of `weights/hyperpeft_lora/feature_names.json`; `weights/hyperpeft_lora/args.json` lists the settings HyperPEFT-LoRA was trained with.
3. Score held-out replies with `evaluation/evaluate_models_10000.py` and run the hidden-state probes with `evaluation/validate_models_10000.py`.
4. Reproduce the paper's tests with the scripts in `analysis/`.

This repository carries the HyperPEFT-LoRA generator and the four shared heads. It does not carry the HyperPEFT-Adapter, HyperPEFT-Bias and HyperPEFT-Prefix generators or the other comparison checkpoints the paper reports.

## What is not released

No post text, usernames, thread identifiers, per-user adapters, or any mapping from keys to identities.

## License

The code, the generator, the shared heads and the data are released under the MIT License (`LICENSE`). The Pythia-160M backbone is EleutherAI's work, distributed under the Apache License 2.0; its text is in `LICENSE-APACHE-2.0.txt`, and `NOTICE` says which files it covers.

## Citation

```bibtex
@inproceedings{walsh_ford_runfola_2027_hyperpeftnet,
  author    = {Tyson Walsh and Trenton W. Ford and Daniel Runfola},
  title     = {{HyperPEFTNet}: Parameter-Efficient Hypernetworks for Persona Synthesis},
  booktitle = {Proceedings of the 32nd International Conference on Intelligent User Interfaces},
  series    = {IUI '27},
  year      = {2027},
  address   = {Helsinki, Finland},
  publisher = {Association for Computing Machinery}
}
```
