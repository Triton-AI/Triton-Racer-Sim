# Triton-Racer-Sim
An autonomous robocar simulation client, designed to work with donkey gym

## Progress
Driving and training pipeline is up and running.
Issue with model loss. Please use donkeycar to train for performance.

## How to Use

### Setup Environment
1. Install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. `conda create -name tritonracer python=3.8`
3. `conda activate tritonracer`
4. Ubuntu 20.04: `pip install pygame==2.0.0.dev10` | Others: `pip install pygame`
5. `pip install docopt tensorflow pillow keras`
6. `conda install scikit-learn`
7. Setup [donkey gym](http://docs.donkeycar.com/guide/simulator/#install) in this environment (omit `conda activate donkey`)


### Drive the Car
At this stage the car lives in car_templates. go to the directory for driving and training

`python manage.py drive`

Use a PS4 joystick.
* Left X axis: steering
* Right Y axis: throttle
* Circle: toggle recording
* Triangle: delete 100 records
* Square: reset the car
* Share: toggle driving mode

Data recorded can be found in data/records_x/

**By default data collection is turned off**

### Train a Model

`python manage.py train --tub data/records_1 --model ./models/pilot.h5` 

* `--tub`: path to the data folder
* `--model`: where you would like to put your model

### Test the Model

`python manage.py drive --model ./models/pilot.h5`

Press "share" to switch between modes:

* Full human control
* Human throttle + AI steering
* Full AI control

## Thoughts