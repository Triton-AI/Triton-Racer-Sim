# Triton-Racer-Sim
An autonomous robocar simulation client, designed to work with [donkey gym](https://github.com/tawnkramer/gym-donkeycar).

![](cover.png)

The project is inspired by, and has taken reference from [donkeycar](https://www.donkeycar.com/).

## Progress
Driving and training pipeline is up and running.

Config added. Use `python manage.py generateconfig` to create one.

## Features
* Speed-based models
* Throttle and steering lock at launch
* Adjustable image resolution in donkey gym
* Ability to reset the car to startline
* Ability to break
* Inter-deployable models with donkeycar (throttle-based models only)
* Compatability with donkeycar parts

## Install

### Set up Environment
1. Install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. `conda create -n tritonracer python=3.8`
3. `conda activate tritonracer`
4. `pip install docopt tensorflow pillow keras opencv-python pygame==2.0.0.dev10`
5. `conda install scikit-learn`
6. Setup [donkey gym](http://docs.donkeycar.com/guide/simulator/#install) in this environment (omit `conda activate donkey` in the original installation procedure)
    1. If you have a donkeycar installation with donkey gym setup, navigate to the donkey gym repository. If not, find a suitable place and `git clone https://github.com/tawnkramer/gym-donkeycar`
    2. Make sure you are still in tritonracer environment `conda activate tritonracer`
    3. `pip install -e .[gym-donkeycar]`

### Download TritonRacer Repo
1. `git clone https://github.com/Triton-AI/Triton-Racer-Sim`
2. Copy `Triton-Racer-Sim/TritonRacerSim/car_templates/` folder to somewhere outside the repository (and optionally rename it). This is your car folder, like "d3" of donkeycar.
3. **IMPORTANT:** Go to your car folder, open manage.py and edit line 11 to be your path to the Triton-Racer-Sim repo in your system `sys.path.append('/home/haoru/Projects/Triton-Racer-Sim/')`

## Manual

Creation of car instances will be supported later.

### Config File
`python manage.py generateconfig` will create a myconfig.json under the car_template. 

**For detailed information about what each setting does: check out `TritonRacerSim/core/config.py` for detailed explanations.**

### Drive the Car

Create a config file, then enter

    python manage.py drive

**IMPORTANT: by default data collection is turned OFF**

Use a PS4 joystick:
* Left X axis: steering
* Right Y axis: throttle
* Right Trigger: breaking
* Circle: toggle recording
* Triangle: delete 100 records
* Square: reset the car
* Share: switch driving mode

Use an XBOX joystick:
* Left X axis: steering
* Right Y axis: throttle
* Right Trigger: breaking
* B: toggle recording
* Y: delete 100 records
* X: reset the car
* Back (below XBOX icon on the left): switch driving mode 

Use a G28 Driving Wheel (oof)
* Steering, throttle and breaking are mapped to the corresponding equipment
* The rest is the same as PS4

Data recorded can be found in data/records_x/

### Before Training

If you have a GPU: install [CUDA support for tensorflow](https://www.tensorflow.org/install/gpu)

Note: If you have CUDA installed, and tensorflow says `Could not load dynamic library 'libcublas.so.10'` and falls back to CPU during training, add these lines to your ~/.bashrc:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64

### Train a Model

`python manage.py train --tub data/records_1 --model ./models/pilot.h5` 

* `--tub`: path to the data folder
* `--model`: where you would like to put your model
* `--transfer`: (optional) which old model you would like to train upon

**IMPORTANT:** `--tub` param is mandatory.

RAM usage (120 * 160 image, 7000 Records): 8GB

VRAM usage (120 * 160 image, 128 batch size): 3GB

**IMPORTANT** If you run into insufficient memory issue, follow [this](https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-18-04/) guide to pre-allocate space for Ubuntu swap file (Allocate more than the size of your physical RAM). Windows users, click [here](https://answers.microsoft.com/en-us/windows/forum/windows_10-performance/how-to-increase-virtual-memory-in-windows-10-a/46dacaf5-15cf-4f5d-9d5a-cba1401ae4c9). Your system may stall momentarily duing training in the first two epoches.


#### Model Types

* `cnn_2d`: Take an image and predict throttle and steering;
* `cnn_2d_speed_as_feature`: Take an image and current speed, and predict throttle and steering
* `cnn_2d_speed_control`: Take an image and predict speed and steering;
* `cnn_2d_full_house`: Take an image, segment of track, and current speed, and predict speed and steering. (Speed prediction is unrelated to current speed)

### Test the Model

`python manage.py drive --model ./models/pilot.h5`

Switch between modes:

* Full human control
* Human throttle + AI steering
* Full AI control

### Migrate Component from Donkeycar, or Write Your Own
How to write your custom component for tritonracer:

1. Subclass the Component class `from TritonRacerSim.components.component import Component`.
2. Define the names of the inputs and outputs for the component in the constructor `super().__init__(self, inputs=['cam/img'], outputs=['ai/steering', 'ai/throttle'], threaded=False)`
3. Implement `step(self, *args)`. Called when the car's main loop iterate through its parts at 20Hz (equivalent to `run()` of donkeycar). Expect args to have the same number of elements as defined in the inputs, and remember to return the same number of outputs as defined in the outputs.
4. Implement `thread_step(self)` if it is a component with its own thread of, for example, polling information from peripherals. thread_step is started before the car starts (equivalent to `update()` of donkeycar). Remember to put a loop, and some delay time inbetween, for the code to run periodically.
5. (Optional) Implement other APIs `onStart(self)`, `onShutdown(self)`, `getName(self)`
6. Add your component in manage.py's `assemble_car()`: `car.addComponent(my_component)`

## Roadmap
Features to come:
1. 3D covolution networks
2. Reinforcement learning
3. Migration to real car
4. ~~Image filtering~~
5. Packaging the software
6. Merging with donkeycar
