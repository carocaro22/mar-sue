# Rock, Paper, Scissors

Relevant code in in programm.py

The code was written using Pyhton 3.12.3

## Files 

**LSTM.ipynb**: explains briefly how the data is trained once using H1 and C1.

**programm.py**: iterates over all (computer, human) pairs and outputs details to a tensorboard.

## Installing

Create Python virtual environment, activate it and install packages

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## (optional) Install tkinter for viweing matplotlib graphics in a window

Picture is also saved as a .png in the root folder, so this step is optional.

```bash
sudo apt install python3-tk
```

## Start tensorboard

```bash
tensorboard --logdir=/.logs/
```

## Run: 

```bash
python3 programm.py
```