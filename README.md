Installation
=============
* Install Mujoco
* Follow installation process of dm_control: https://github.com/deepmind/dm_control

Scripts
=======
CMU Simulation
--------------
Imports amc files from the CMU Mocap Dataset: http://mocap.cs.cmu.edu/.
Visualizes animation of motion in Mujoco and outputs a series of data features in 3 coordinate systems.
The model used in this script follows the humanoid_CMU.xml model from the dm_control repo.

The following files are created by this script representing features in the world (global), center of mass (thorax), and parent body reference frames:
* `features/<filename>_features_world.txt`
* `features/<filename>_features_com.txt`
* `features/<filename>_features_parent.txt`

Each file contains the following features:
* Freejoint data: position (3), orientation (9), linear velocity (3), angular velocity (3), linear acceleration (3), angular acceleration (3)
* Per DOF joint data: position (3), orientation (3 x 3), rotation axis (3), rotation angle (1), velocity (1), acceleration (1)

Input arguments:
* `--filename`: single .amc input file
* `--dir`: directory containing all .amc files (searches directory recursively)
* `--filter`: tsv file containing names of filtered amc files
* `--max_num_frames`: maximum number of frames to simulate
* `--simulate`: displays simulation

Example of execution:
(for single file)  
`python scripts/simulate_cmu.py --filename=cmu-data/filename.amc --max_num_frames=10000 --simulate`

OR (for all amc files)  
`python scripts/simulate_cmu.py --dir=cmu-data/all_asfamc/subjects --max_num_frames=10000`

PUSH Simulation
---------------
Imports npy files from the PUSH dataset and visualizes the motion in Mujoco.
The script outputs joint positions for the Left and Right Shank and Thigh (4 positions in total) per frame.
The output file is named `<filename>_positions.npy` and is placed under the IK_cut_spline directory.

The output file is in the following format in the global frame of reference:
* Per frame: position of left shank (ankle joint) (3), position of right shank (ankle joint) (3), position of left thigh (knee joint) (3), position of right thigh (knee joint) (3)

Input arguments:
* `--filename`: single .pyn input file
* `--dir`: directory containing all .pyn files (searches directory recursively)
* `--maxframe`: maximum number of frames to simulate
* `--simulate`: displays simulation

Example of execution:
(for single file)  
`python scripts/simulate_PUSH.py --filename=/home/kelly/Documents/motion-sim/IK_cut_spline/LegAbductionSet01.npy --maxframe=3000 --simulate=true`

PUSH IMU Acceleration Generation
--------------------------------
This script takes input the ankle and knee joint positions from the above script and generates acceleration data for each joint.
The script applies Savgol filter to smoothen the position, velocity and acceleration from differentiation calculations and compares the result to a smoothened IMU acceleration measurement from sensors of the same dataset. The optimal rotation matrix for mapping the differentiated acceleration to the IMU acceleration is calculated using Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm).

The output of this script includes 2 Figures and 1 output file for each joint:
* Figure 1 - 6 subplots
    * Raw Position (XYZ)
    * Position with Savgol Filter (XYZ)
    * Raw Differentiated Velocity (XYZ)
    * Velocity with Savgol Filter (XYZ)
    * Raw Differentiated Acceleration (XYZ)
    * Acceleration with Savgol Filter (XYZ)
* Figure 2 - 3 subplots
    * Raw IMU Acceleration (XYZ)
    * IMU Acceleration with Savgol Filter (XYZ)
    * Differentiated Acceleration with Savgol Filter with Rotation Transformation (XYZ)
* Output File - 1 per joint (4 in total) named `<filename>_<joint_name>.npy` under IK_cut_spline directory
    * XYZ coordinates of differentiated acceleration after Savgol Filter and with rotation matrix applied

Example of execution:
`python scripts/PUSH_acceleration.py --filename=/home/kelly/Documents/motion-sim/IK_cut_spline/RunNormalSpeed1.npy`

Other Scripts
-------------
* utdmhad_acceleration - simulate UTD-MHAD dataset (http://www.utdallas.edu/~kehtar/UTD-MHAD.html) IMU Acceleration data based on differentiation and fitlering of joint positions
* simulate_NTU - attempt to visualize NTU RGB+D Action Recognition Dataset (https://github.com/shahroudy/NTURGB-D) into Mujoco



Other resources:
----------------

Slides:
https://docs.google.com/presentation/d/1DdoyGRLiAPB51EXB8_uJ7wif-bwUNkAy8J-ltemoeUE/edit?usp=sharing

Google Drive:
https://drive.google.com/drive/folders/1r7CAxZYcsMVtZUxZtLA_FjZtFHNvNRQy?usp=sharing