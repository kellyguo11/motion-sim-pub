Installation:
-------------
* Install Mujoco
* Follow installation process of dm_control: https://github.com/deepmind/dm_control

Conversion Script:
------------------
From root:  

(for single file)  
`python scripts/simulate_cmu.py --filename=cmu-data/filename.amc --max_num_frames=10000 --simulate`

OR (for all amc files)  
`python scripts/simulate_cmu.py --dir=cmu-data/all_asfamc/subjects --max_num_frames=10000`

Input arguments:
----------------
* `--filename`: single .amc input file
* `--dir`: directory containing all .amc files (searches directory recursively)
* `--max_num_frames`: maximum number of frames to simulate
* `--simulate`: displays simulation
* `--noise`: adds noise to motion

Output File:
------------
From root: <amc_file_name>_joints.csv

Output Columns:
* Frame Index: 0-based index of frame number
* Joint Index: 0-based index of all joints in model (not including the 0th root freejoint)
* Joint Angle: angle in rads relative to fixed body axis of each joint (qpos)
* Joint Velociy: velocity of each joint (qvel)
* Local Joint Axis: fixed body axis for each joint
* Global Joint Axis: Computed joint axis in the global frame from joint angle 