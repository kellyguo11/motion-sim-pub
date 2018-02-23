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
* `--filter`: tsv file containing names of filtered amc files
* `--max_num_frames`: maximum number of frames to simulate
* `--simulate`: displays simulation
* `--noise`: adds noise to motion

Slides:
https://docs.google.com/presentation/d/1DdoyGRLiAPB51EXB8_uJ7wif-bwUNkAy8J-ltemoeUE/edit?usp=sharing

Google Drive:
https://drive.google.com/drive/folders/1r7CAxZYcsMVtZUxZtLA_FjZtFHNvNRQy?usp=sharing