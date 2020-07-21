# DRP-attack
DRP-attack


### Requierment

- Python 3.7.6 or higher
- Ubuntu 18.04
- CUDA 10.1

### Installation

- Clone this repository. Assume it is downloaded to 
- Install dependencies `pip install -r requirements.txt`
- Clone OpenPilot v0.6.6 under DRP attack directory, i.e., `/foobar/DRP-attack/openpilot`.
    -  `git clone https://github.com/commaai/openpilot -b v0.6.6` 
- Install capnp
    - `cd /foobar/DRP-attack/openpilot/phonelibs`
    - `sudo sh install_capnp.sh`
- Compile messaging
    - `cd /foobar/DRP-attack/openpilot/selfdrive/messaging`
    - `PYTHONPATH=$PYTHONPATH:/home/takamisato/lab_gpu/DRP-attack/openpilot/  make`
- Compile libmpc
    - `cd /foobar/DRP-attack/openpilot/selfdrive/controls/lib/lateral_mpc`
    - `find ./ -name "*.o" | xargs rm`
    - `make all`

### Usage

This repository has a highway scenarios.

```bash
### attack to left
python run_patch_generation.py data/scenarios/highway/sc1/config_left.json
### attakc to right
python run_patch_generation.py data/scenarios/highway/sc1/config_right.json
```

For more detail, please check Tutorial notebook.