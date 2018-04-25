# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Demonstration of amc parsing for CMU mocap database.

To run the demo, supply a path to a `.amc` file:

    python mocap_demo --filename='path/to/mocap.amc'

CMU motion capture clips are available at mocap.cs.cmu.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Internal dependencies.

from absl import app
from absl import flags

from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import os, sys
import copy
import transforms3d

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'txt file to be converted.')
flags.DEFINE_string('dir', None, 'directory containing all txt files.')
flags.DEFINE_boolean('simulate', False, 'Simulate results.')

JOINTS_ORDER = ["lfemurrz", "lfemurry", "lfemurrx", "ltibiarx", "lfootrz", "lfootrx", "ltoesrx" ,
                "rfemurrz", "rfemurry", "rfemurrx", "rtibiarx", "rfootrz" , "rfootrx" , "rtoesrx" ,
                "lowerbackrz", "lowerbackry", "lowerbackrx", "upperbackrz", "upperbackry", "upperbackrx", "thoraxrz",
                "thoraxry", "thoraxrx", "lowerneckrz", "lowerneckry", "lowerneckrx", "upperneckrz", "upperneckry",
                "upperneckrx", "headrz", "headry", "headrx", "lclaviclerz", "lclaviclery", "lhumerusrz", 
                "lhumerusry", "lhumerusrx", "lradiusrx", "lwristry", "lhandrz" , "lhandrx" , "lfingersrx",
                "lthumbrz", "lthumbrx", "rclaviclerz", "rclaviclery", "rhumerusrz", "rhumerusry", "rhumerusrx", 
                "rradiusrx", "rwristry", "rhandrz" , "rhandrx" , "rfingersrx", "rthumbrz", "rthumbrx"]

INIT_QPOS = [-20, 0, 0, 0, 0, 0, 0,
            20, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 85, 
            0, -30, 0, 0, 0, 0, 0,
            0, 0, 0, 0, -85, 0, -30, 
            0, 0, 0, 0, 0, 0, 0]

'''
Configuration of 25 body joints in our dataset. The labels
of the joints are: 1-base of the spine 2-middle of the spine
3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8-
left hand 9-right shoulder 10-right elbow 11-right wrist 12-
right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17-
right hip 18-right knee 19-right ankle 20-right foot 21-spine 22-
tip of the left hand 23-left thumb 24-tip of the right hand 25-
right thumb
'''

JOINT_MAPPING = dict()
JOINT_MAPPING[1] = ["lowerbackrx", "lowerbackry", "lowerbackrz"]
JOINT_MAPPING[2] = ["upperbackrx", "upperbackry", "upperbackrz"]
JOINT_MAPPING[3] = ["upperneckrx", "upperneckry", "upperneckrz"]
JOINT_MAPPING[4] = ["headrx", "headry", "headrz"]
JOINT_MAPPING[5] = ["lhumerusrx", "lhumerusry", "lhumerusrz"]
JOINT_MAPPING[6] = ["lradiusrx"]
JOINT_MAPPING[7] = ["lwristry"]
JOINT_MAPPING[8] = ["lhandrx", "lhandrz"]
JOINT_MAPPING[9] = ["rhumerusrx", "rhumerusry", "rhumerusrz"]
JOINT_MAPPING[10] = ["rradiusrx"]
JOINT_MAPPING[11] = ["rwristry"]
JOINT_MAPPING[12] = ["rhandrx", "rhandrz"]
JOINT_MAPPING[13] = ["lfemurrx", "lfemurry", "lfemurrz"]
JOINT_MAPPING[14] = ["ltibiarx"]
JOINT_MAPPING[15] = ["lfootrx", "lfootrz"]
JOINT_MAPPING[16] = ["ltoesrx"]
JOINT_MAPPING[17] = ["rfemurrx", "rfemurry", "rfemurrz"]
JOINT_MAPPING[18] = ["rtibiarx"]
JOINT_MAPPING[19] = ["rfootrx", "rfootrz"]
JOINT_MAPPING[20] = ["rtoesrx"]
JOINT_MAPPING[21] = ["lowerneckrx", "lowerneckry", "lowerneckrz"]
JOINT_MAPPING[22] = ["lfingersrx"]
JOINT_MAPPING[23] = ["lthumbrx", "lthumbrz"]
JOINT_MAPPING[24] = ["rfingersrx"]
JOINT_MAPPING[25] = ["rthumbrx", "rthumbrz"]

PARENT_JOINT_MAPPING = [1, 1, 2, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]

def normalize(v):
  norm = np.linalg.norm(v, ord=1)
  if norm == 0:
      norm = np.finfo(v.dtype).eps
  return v/norm

def isLeft(j):
  return (j+1 in range(5, 9) or j+1 in range(15, 19))

def parsedata(filename, sim):
  file = open(filename, 'r')
  lines = file.read().split('\n')

  pos_list = [] # [[list of joints for the frame])

  num_frames = int(lines[0])
  skip_lines = 0
  for i in range(num_frames):
    num_bodies = lines[1+skip_lines]
    dont_care = lines[2+skip_lines]
    num_joints = int(lines[3+skip_lines])

    frame_info = []

    for j in range(num_joints):
      joint_info = lines[j+4+skip_lines].split(" ")
      pos_x = float(joint_info[0])
      pos_y = float(joint_info[1])
      pos_z = float(joint_info[2])
      ori_w = float(joint_info[7])
      ori_x = float(joint_info[8])
      ori_y = float(joint_info[9])
      ori_z = float(joint_info[10])

      frame_info.append([pos_x, pos_y, pos_z, ori_w, ori_x, ori_y, ori_z])

    skip_lines += (3 + num_joints)
    pos_list.append(frame_info)

  env = humanoid_CMU.stand()

  max_frame = num_frames

  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  joints = dict()

  for i in range(int(max_frame)):
    with env.physics.reset_context():
      for j in range(len(pos_list[i])):
        joint_names = JOINT_MAPPING[j+1]
        joint_index = env.physics.model.name2id(joint_names[0], 'joint')

        joint_quat = [pos_list[i][j][3], pos_list[i][j][4], pos_list[i][j][5], pos_list[i][j][6]]
        angles = transforms3d.euler.quat2euler(joint_quat)

        #parent_quat = [pos_list[i][0][3], pos_list[i][0][4], pos_list[i][0][5], pos_list[i][0][6]]
        parent_quat = [pos_list[i][PARENT_JOINT_MAPPING[j] - 1][3], pos_list[i][PARENT_JOINT_MAPPING[j] - 1][4], pos_list[i][PARENT_JOINT_MAPPING[j] - 1][5], pos_list[i][PARENT_JOINT_MAPPING[j] - 1][6]]
        parent_angles = transforms3d.euler.quat2euler(parent_quat)

        relative_angles = np.subtract(angles, parent_angles)

        if j in range(12, 14) or j in range(16, 18):
          print(np.degrees(angles))
          print(np.degrees(relative_angles))

          for name in joint_names:
            #if isLeft(j):
            if name[-2:] == 'rz':
              joint_index = env.physics.model.name2id(name, 'joint')
              env.physics.data.qpos[joint_index + 6] = relative_angles[0]

            elif name[-2:] == 'rx':
              joint_index = env.physics.model.name2id(name, 'joint')
              env.physics.data.qpos[joint_index + 6] = relative_angles[1]

            elif name[-2:] == 'ry':
              joint_index = env.physics.model.name2id(name, 'joint')
              env.physics.data.qpos[joint_index + 6] = relative_angles[2]

            # else:
            #   if name[-2:] == 'rz':
            #     joint_index = env.physics.model.name2id(name, 'joint')
            #     env.physics.data.qpos[joint_index + 6] = -relative_angles[1]

            #   elif name[-2:] == 'rx':
            #     joint_index = env.physics.model.name2id(name, 'joint')
            #     env.physics.data.qpos[joint_index + 6] = -relative_angles[2]

            #   elif name[-2:] == 'ry':
            #     joint_index = env.physics.model.name2id(name, 'joint')
            #     env.physics.data.qpos[joint_index + 6] = relative_angles[0]

      env.physics.step()
      print("Processing frame " + str(i))

    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

  file.close()

  plt.figure(2)
  tic = time.time()
  for i in range(max_frame):
    if i == 0:
      img = plt.imshow(video[i])
    else:
      img.set_data(video[i])
    toc = time.time()
    clock_dt = toc - tic
    tic = time.time()
    # Real-time playback not always possible as clock_dt > .03
    plt.pause(np.maximum(0.01, .03 - clock_dt))  # Need min display time > 0.0.
    plt.draw()
  plt.waitforbuttonpress()


def main(unused_argv):
  if FLAGS.dir:
    for filename in glob.iglob(FLAGS.dir):
      parsedata(filename, FLAGS.simulate)
  elif FLAGS.filename:
    parsedata(FLAGS.filename, FLAGS.simulate)
  else:
    print("Please provide input source.")

if __name__ == '__main__':
  #flags.mark_flag_as_required('filename')
  app.run(main)
