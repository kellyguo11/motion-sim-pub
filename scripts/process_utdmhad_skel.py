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

# bodies that joints are attached to (in order of above)
JOINT_BODIES = ["lfemur", "lfemur", "lfemur", "ltibia", "lfoot", "lfoot", "ltoes",
                "rfemur", "rfemur", "rfemur", "rtibia", "rfoot", "rfoot", "rtoes",
                "lowerback", "lowerback", "lowerback", "upperback", "upperback", "upperback", "thorax",
                "thorax", "thorax", "lowerneck", "lowerneck", "lowerneck", "upperneck", "upperneck",
                "upperneck", "head", "head", "head", "lclavicle", "lclavicle", "lhumerus", 
                "lhumerus", "lhumerus", "lradius", "lwrist", "lhand", "lhand", "lfingers",
                "lthumb", "lthumb", "rclavicle", "rclavicle", "rclavicle", "rhumerus", "rhumerus",
                "rhumerus", "rradius", "rwrist", "rhand", "rhand", "rfingers", "rthumb", "rthumb"]

'''
1 Base of the spine - lowerback(rz, ry, rx)
2 Middle of the spine - upperback(rz, ry, rx)
3 Neck - upperneck(rz, ry, rx)
4 Head - head(rz, ry, rx)
5 Left shoulder - lhumerus(rz, ry, rx)
6 Left elbow - lradius(rx)
7 Left wrist - lwrist(ry)
8 Left hand - lhand(rz, rx)
9 Right shoulder - rhumerus(rz, ry, rx)
10 Right elbow - rradius(rx)
11 Right wrist - rwrist(ry)
12 Right hand - rhand(rz, rx)
13 Left hip - lfemur(rz, ry, rx)
14 Left knee - ltibia(rx)
15 Left ankle - lfoot(rz, rx)
16 Left foot - ltoes(rx)
17 Right hip - rfemur(rz, ry, rx)
18 Right knee - rtibia(rx)
19 Right ankle - rfoot(rz, rx)
20 Right foot - rtoes(rx)
21 Spine at the shoulder - lowerneck(rz, ry, rx)
22 Tip of the left hand - lfingers(rx)
23 Left thumb - lthumb(rz, rx)
24 Tip of the right hand - rfingers(rx)
25 Right thumb - rthumb(rz, rx)
'''

JOINT_MAPPING = dict()
JOINT_MAPPING[1] = ["lowerbackrz", "lowerbackry", "lowerbackrx"]
JOINT_MAPPING[2] = ["upperbackrz", "upperbackry", "upperbackrx"]
JOINT_MAPPING[3] = ["upperneckrz", "upperneckry", "upperneckrx"]
JOINT_MAPPING[4] = ["headrx", "headry", "headrx"]
JOINT_MAPPING[5] = ["lhumerusrz", "lhumerusry", "lhumerusrx"]
JOINT_MAPPING[6] = ["lradiusrx"]
JOINT_MAPPING[7] = ["lwristry"]
JOINT_MAPPING[8] = ["lhandrz", "lhandrx"]
JOINT_MAPPING[9] = ["rhumerusrz", "rhumerusry", "rhumerusrx"]
JOINT_MAPPING[10] = ["rradiusrx"]
JOINT_MAPPING[11] = ["rwristry"]
JOINT_MAPPING[12] = ["rhandrz", "rhandrx"]
JOINT_MAPPING[13] = ["lfemurrz", "lfemurry", "lfemurrx"]
JOINT_MAPPING[14] = ["ltibiarx"]
JOINT_MAPPING[15] = ["lfootrz", "lfootrx"]
JOINT_MAPPING[16] = ["ltoesrx"]
JOINT_MAPPING[17] = ["rfemurrz", "rfemurry", "rfemurrx"]
JOINT_MAPPING[18] = ["rtibiarx"]
JOINT_MAPPING[19] = ["rfootrz", "rfootrx"]
JOINT_MAPPING[20] = ["rtoesrx"]
JOINT_MAPPING[21] = ["lowerneckrz", "lowerneckry", "lowerneckrx"]
JOINT_MAPPING[22] = ["lfingersrx"]
JOINT_MAPPING[23] = ["lthumbrz", "lthumbrx"]
JOINT_MAPPING[24] = ["rfingersrx"]
JOINT_MAPPING[25] = ["rthumbrz", "rthumbrx"]


J = [[1, 2, 3, 4, 5, 6, 7, 7, 3, 10, 11, 12, 13, 14, 3, 16, 17, 18, 19, 19, 11, 22, 23, 24],
     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]];

def normalize(v):
  norm = np.linalg.norm(v, ord=1)
  if norm == 0:
      norm = np.finfo(v.dtype).eps
  return v/norm

def parsedata(filename, sim):
  file = open(filename, 'r')
  lines = file.read().split('\n')

  pos_list = dict() # dict(joint_id, [list of positions for all frames])
  frame_count = 0
  num_joints = 0

  init_pos = []

  j = 1
  # split by joints
  for joints in lines:
    if joints != '':
      num_joints += 1
      # split by frames
      frames = joints.split(',')
      if j not in pos_list.keys():
        pos_list[j] = []
      # add position of each joint per frame
      for f in frames:
        positions = f.split(' ')
        positions = normalize(np.array([float(positions[0]), float(positions[2]), float(positions[1])]))
        pos_list[j].append(positions)
        if j == 1:
          frame_count += 1
      j += 1

  # build initial joint positions (for frame 0)
  for i in range(1, num_joints + 1):
    init_pos.append(pos_list[i][0])

  env = humanoid_CMU.stand()

  max_frame = frame_count

  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  #createjointcsv(output_filename)
  #createbodiescsv(os.path.basename(filename).split('.')[0] + "_bodies.csv")
  output_world = []
  output_com = []
  output_parent = []
  joints = dict()
  bodies = dict()

  default_angle = []

  for i in range(int(max_frame)):
    with env.physics.reset_context():
      for j in range(1, num_joints + 1):
        joint_names = JOINT_MAPPING[j]
        cur_pos = pos_list[j][i]

        joint_index = env.physics.model.name2id(joint_names[0], 'joint')
        body_index = env.physics.model.name2id(JOINT_BODIES[joint_index], 'body')

        body_pos = env.physics.data.xpos[body_index]
        body_rot = env.physics.data.xmat[body_index].reshape(3, 3)

        #joint_2_joint_vec = normalize(np.subtract(np.array(pos_list[j][i]), np.array(pos_list[J[0][j - 2]][i])))
        joint_pos = pos_list[j][i]

        # calculate initial angle from default mujoco model
        if i == 1:
          default_joint_pos = body_pos

          # convert positions to body-frame reference
          body_frame_pos_prev = np.array([0.0, 0.0, 0.0])
          body_frame_pos_cur = np.matmul(np.subtract(joint_pos, body_pos), body_rot)

          angle_init = [0, 0, 0]

          # calculate rotation angle around each axis
          for name in joint_names:
            if name[-2:] == 'rz':
              #project onto xy plane
              normal = np.cross(body_rot[0,:], body_rot[1,:])
              projection = np.subtract(body_frame_pos_cur, np.multiply(np.dot(body_frame_pos_cur, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              projection_prev = np.array([0.0, 0.0, 1.0])
              angle = np.arccos(np.dot(projection_prev, projection)/(np.linalg.norm(projection_prev) * np.linalg.norm(projection)))
              rad = np.radians(angle)
              joint_index = env.physics.model.name2id(name, 'joint')
              #env.physics.data.qpos[joint_index + 6] = rad
              #angle_init[0] = rad
            elif name[-2:] == 'rx':
              #project onto yz plane
              normal = np.cross(body_rot[1,:], body_rot[2,:])
              projection = np.subtract(body_frame_pos_cur, np.multiply(np.dot(body_frame_pos_cur, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              projection_prev = np.array([1.0, 0.0, 0.0])
              angle = np.arccos(np.dot(projection_prev, projection)/(np.linalg.norm(projection_prev) * np.linalg.norm(projection)))
              rad = np.radians(angle)
              joint_index = env.physics.model.name2id(name, 'joint')
              #env.physics.data.qpos[joint_index + 6] = rad
              #angle_init[1] = rad
            elif name[-2:] == 'ry':
              #project onto xz plane
              normal = np.cross(body_rot[0,:], body_rot[2,:])
              projection = np.subtract(body_frame_pos_cur, np.multiply(np.dot(body_frame_pos_cur, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              projection_prev = np.array([0.0, 1.0, 0.0])
              angle = np.arccos(np.dot(projection_prev, projection)/(np.linalg.norm(projection_prev) * np.linalg.norm(projection)))
              rad = np.radians(angle)
              joint_index = env.physics.model.name2id(name, 'joint')
              #env.physics.data.qpos[joint_index + 6] = rad
              #angle_init[2] = rad
          default_angle.append(angle_init)
        elif i > 1 and j > 1:
          #init_joint_vec = normalize(np.subtract(np.array(init_pos[j-1]), np.array(init_pos[J[0][j - 2]-1])))
          init_joint_vec = init_pos[j-1]

          # convert positions to body-frame reference
          body_frame_pos_prev = np.matmul(np.subtract(init_joint_vec, body_pos), body_rot)
          body_frame_pos_cur = np.matmul(np.subtract(joint_pos, body_pos), body_rot)

          diff = np.subtract(body_frame_pos_cur, body_frame_pos_prev)
          if np.array_equal(diff, np.array([0.0, 0.0, 0.0])):
            continue

          # calculate rotation angle around each axis
          for name in joint_names:
            if name[-2:] == 'rz':
              #project onto xy plane
              normal = np.cross(body_rot[0,:], body_rot[1,:])
              projection = np.subtract(body_frame_pos_cur, np.multiply(np.dot(body_frame_pos_cur, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              projection_prev = np.subtract(body_frame_pos_prev, np.multiply(np.dot(body_frame_pos_prev, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              angle = np.arccos(np.dot(projection_prev, projection)/(np.linalg.norm(projection_prev) * np.linalg.norm(projection)))
              rad = np.radians(angle*360)
              joint_index = env.physics.model.name2id(name, 'joint')
              env.physics.data.qpos[joint_index + 6] = rad + default_angle[j-1][0]
            elif name[-2:] == 'rx':
              #project onto yz plane
              normal = np.cross(body_rot[1,:], body_rot[2,:])
              projection = np.subtract(body_frame_pos_cur, np.multiply(np.dot(body_frame_pos_cur, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              projection_prev = np.subtract(body_frame_pos_prev, np.multiply(np.dot(body_frame_pos_prev, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              angle = np.arccos(np.dot(projection_prev, projection)/(np.linalg.norm(projection_prev) * np.linalg.norm(projection)))
              rad = np.radians(angle*360)
              joint_index = env.physics.model.name2id(name, 'joint')
              env.physics.data.qpos[joint_index + 6] = rad + default_angle[j-1][1]
            elif name[-2:] == 'ry':
              #project onto xz plane
              normal = np.cross(body_rot[0,:], body_rot[2,:])
              projection = np.subtract(body_frame_pos_cur, np.multiply(np.dot(body_frame_pos_cur, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              projection_prev = np.subtract(body_frame_pos_prev, np.multiply(np.dot(body_frame_pos_prev, normal), normal)/(np.linalg.norm(normal)*np.linalg.norm(normal)))
              angle = np.arccos(np.dot(projection_prev, projection)/(np.linalg.norm(projection_prev) * np.linalg.norm(projection)))
              rad = np.radians(angle*360)
              joint_index = env.physics.model.name2id(name, 'joint')
              env.physics.data.qpos[joint_index + 6] = rad + default_angle[j-1][2]

      env.physics.step()
      print("Processing frame " + str(i))
      #outputbodies(os.path.basename(filename).split('.')[0] + "_bodies.csv", env.physics, i)
      # bodies = calcBodyFrames(env.physics)
      # #joints = outputjoints2csv(output_filename, i, env.physics, bodies, joints)
      # joints, frame_data_world, frame_data_com, frame_data_parent = buildFeatures(i, env.physics, bodies, joints)
      # output_world.append(frame_data_world)
      # output_com.append(frame_data_com)
      # output_parent.append(frame_data_parent)
      #outputcppcsv(os.path.basename(filename).split('.')[0] + "_cpp", env.physics)

    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

    #outputstate2csv(filename + "_state.csv", converted)
  # writeOutput(file_prefix + "_features_world.txt", output_world)
  # writeOutput(file_prefix + "_features_com.txt", output_com)
  # writeOutput(file_prefix + "_features_parent.txt", output_parent)

  if sim:
    #visualizeJoint(49, joints)

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
