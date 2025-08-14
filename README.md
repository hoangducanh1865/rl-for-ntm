# Reinforcement Learning-based Topic Model

## To create virtual environment, run:

```
python3.10 -m venv venv
```

## To get into created virtual environment, run:

```
source venv/bin/activate
```

## To install required libraries, run:

```
pip3 install -r requirements.txt
```

## To train, run:

```
python3 main.py 'src/datasets/pickles/20newsgroups_mwl3.pkl'
```

## To experiment, run: 

```
python3 run_experiments.py my_experiment 5 --meta_seed 42
```