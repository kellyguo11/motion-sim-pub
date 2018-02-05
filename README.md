Installation:
-------------
Install Mujoco
Follow installation process of dm_control: https://github.com/deepmind/dm_control

Conversion Script:
------------------
from root:
(for single file)
python scripts/simulate_cmu.py --filename=cmu-data/filename.amc --max_num_frames=10000 --simulate
OR (for all amc files)
python scripts/simulate_cmu.py --dir=cmu-data/all_asfamc/subjects --max_num_frames=10000

Input arguments:
----------------
--filename: single .amc input file
--dir: directory containing all .amc files (searches directory recursively)
--max_num_frames: maximum number of frames to simulate
--simulate: displays simulation
--noise: adds noise to motion

Output File:
------------
from root: <amc_file_name>_joints.csv