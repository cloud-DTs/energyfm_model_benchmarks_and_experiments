# EnergyFM Benchmarks

Compare Chronos 2, Chronicle, Toto 2.0 22M and Chronos 2 Small, then run the E1 weather-context, E2 distillation and E3 continual-learning experiments.

## Setup

Use Python 3.13 on macOS 15 or later on Apple silicon, Linux x86-64, or Windows x64. Open a terminal in the repository root.

**macOS and Linux**

```sh
python3.13 -m venv .venv
. .venv/bin/activate
```

**Windows Command Prompt**

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

**All three platforms**

Install the CPU build of PyTorch and then the project with pip. [PyTorch CPU packages](https://download.pytorch.org/whl/cpu/torch/)

```sh
python -m pip install --no-deps torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[cpu]"
```

Download the pinned model versions and the VEA and UCI datasets.

```sh
python -m energyfm fetch models chronos chronicle toto chronos-small
python -m energyfm fetch data vea uci
```

The prepared E1 inputs are included in `data/e1/`. Model weights and the VEA and UCI archives are downloaded on first setup.

## Try all four benchmarks

Run these commands in the same virtual environment.

```sh
python -m energyfm benchmark --limit 2
python -m energyfm e1 --limit 2
python -m energyfm e2 compare --smoke --dataset vea
python -m energyfm e2 compare --smoke --dataset uci
python -m energyfm e3 replay --smoke --model toto --seeds 42
python -m energyfm e3 replay --smoke --model chronos-small --seeds 42
```

The first command compares the four base models on the same windows. E1 compares forecasts with and without archived weather. E2 runs a small training comparison for each dataset. E3 screens one VEA profile per student and replays an update when triggered. Results are written to `runs/`. Reference results from earlier runs are in `reference/`.

## Run the full experiments

Generate the E2 teacher targets once per dataset. On later runs, start with the corresponding `e2 compare` command.

```sh
python -m energyfm benchmark
python -m energyfm e1

python -m energyfm e2 prepare-targets --dataset vea --kind quantile
python -m energyfm e2 compare --dataset vea
python -m energyfm e2 prepare-targets --dataset uci --kind quantile
python -m energyfm e2 compare --dataset uci

python -m energyfm e3 replay --model toto
python -m energyfm e3 replay --model chronos-small
```

The other E2 experiments are listed under `python -m energyfm e2 --help`.
