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
from scipy.signal import savgol_filter
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn import gaussian_process
import glob
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'amc file to be converted.')
flags.DEFINE_integer('max_num_frames', 90,
                     'Maximum number of frames for plotting/playback')
#flags.DEFINE_string('output_state_csv', None, 'csv file to contain output of state.')
#flags.DEFINE_string('output_joints_csv', None, 'csv file to contain output of joints.')
flags.DEFINE_string('dir', None, 'directory containing all amc files.')
flags.DEFINE_boolean('simulate', False, 'Simulate results.')
flags.DEFINE_boolean('noise', False, 'Add noise to motion.')

JOINTS_ORDER = ["lfemurrz", "lfemurry", "lfemurrx", "ltibiarx", "lfootrz", "lfootrx", "ltoesrx" ,
                "rfemurrz", "rfemurry", "rfemurrx", "rtibiarx", "rfootrz" , "rfootrx" , "rtoesrx" ,
                "lowerbackrz", "lowerbackry", "lowerbackrx", "upperbackrz", "upperbackry", "upperbackrx", "thoraxrz",
                "thoraxry", "thoraxrx", "lowerneckrz", "lowerneckry", "lowerneckrx", "upperneckrz", "upperneckry",
                "upperneckrx", "headrz", "headry", "headrx", "lclaviclerz", "lclaviclery", "lhumerusrz", 
                "lhumerusry", "lhumerusrx", "lradiusrx", "lwristry", "lhandrz" , "lhandrx" , "lfingersrx",
                "lthumbrz", "lthumbrx", "rclaviclerz", "rclaviclery", "rhumerusrz", "rhumerusry", "rhumerusrx", 
                "rradiusrx", "rwristry", "rhandrz" , "rhandrx" , "rfingersrx", "rthumbrz", "rthumbrx"]

JOINT_BODIES = ["lfemur", "lfemur", "lfemur", "ltibia", "lfoot", "lfoot", "ltoes",
                "rfemur", "rfemur", "rfemur", "rtibia", "rfoot", "rfoot", "rtoes",
                "lowerback", "lowerback", "lowerback", "upperback", "upperback", "upperback", "thorax",
                "thorax", "thorax", "lowerneck", "lowerneck", "lowerneck", "upperneck", "upperneck",
                "upperneck", "head", "head", "head", "lclavicle", "lclavicle", "lhumerus", 
                "lhumerus", "lhumerus", "lradius", "lwrist", "lhand", "lhand", "lfingers",
                "lthumb", "lthumb", "rclavicle", "rclavicle", "rclavicle", "rhumerus", "rhumerus",
                "rhumerus", "rradius", "rwrist", "rhand", "rhand", "rfingers", "rthumb", "rthumb"]

bodies = dict()

def outputstate2csv(output_state_csv, converted):
  with open(output_state_csv, 'w') as output:
    writer = csv.writer(output, delimiter=',')
    #len(qpos) = nq (# of position coordinates)
    writer.writerow(['qpos (position)', 'qvel (velocity)', 'time'])
    len_qpos = len(converted.qpos)
    len_qvel = len(converted.qvel)
    len_time = len(converted.time)
    count = 0
    for qpos, qvel, time in list(itertools.zip_longest(converted.qpos, converted.qvel, converted.time)):
      pos = 'None'
      vel = 'None'
      t = 'None'
      if count < len_qpos:
        pos = ' '.join([str(i) for i in qpos])
      if count < len_qvel:
        vel = ' '.join([str(i) for i in qvel])
      if count < len_time:
        t = time
      writer.writerow([pos, vel, t])
      count = count + 1

def outputjoints2csv(output_joints_csv, frame, physics):

  joint_angle = physics.position()
  joint_vel = physics.velocity()
  joint_axis = physics.data.xaxis

  if output_joints_csv:
    with open(output_joints_csv, 'a') as output:
      writer = csv.writer(output, delimiter=',')

      #output freejoint info
      row = [str(frame), 0, 'root', [joint_angle[0:3].tolist(), joint_angle[3:7].tolist()], '', [joint_vel[0:3].tolist(), joint_vel[3:6].tolist()], 
        joint_axis[0], bodies['root'][0], bodies['root'][1], bodies['root'][2], bodies['root'][3], bodies['root'][4], bodies['root'][5]]
      writer.writerow(row)

      # remove freejoint 'root'
      joint_angle = physics.position()[7:]
      joint_vel = physics.velocity()[6:]
      joint_axis = physics.data.xaxis[1:]

      for i in range(len(JOINTS_ORDER)):
        joint_name = physics.model.name2id(JOINTS_ORDER[i], 'joint')
        row = [str(frame), joint_name, JOINTS_ORDER[i]]
        
        row.append(joint_angle[i])
        row.append(np.rad2deg(joint_angle[i]))
        row.append(joint_vel[i])
        row.append(joint_axis[i])
        row.append(bodies[JOINT_BODIES[i]][0])
        row.append(bodies[JOINT_BODIES[i]][1])
        row.append(bodies[JOINT_BODIES[i]][2])
        row.append(bodies[JOINT_BODIES[i]][3])
        row.append(bodies[JOINT_BODIES[i]][4])
        row.append(bodies[JOINT_BODIES[i]][5])

        writer.writerow(row)

def createjointcsv(output_joints_csv):
  with open(output_joints_csv, 'w') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerow(['frame index', 'joint index', 'joint name', 'joint angle/position', 'joint angle (deg)', 'joint velocity', 
      'cartesian joint axis', 'World Position', 'World Orientation', 'COM Position', 'COM Orientation',
      'Parent Position', 'Parent Orientation'])

def outputcppcsv(physics):
  qpos = physics.data.qpos

  with open('cpp_jnt_pos', 'a') as output:
    writer = csv.writer(output, delimiter=',')
    for i in range(len(qpos)):
      writer.writerow([qpos[i]])

def outputbodies(physics, frame):
  with open('bodies.csv', 'a') as output:
    writer = csv.writer(output, delimiter=',')
    for i in range(physics.model.nbody):
      row = [frame]
      row.append(i)
      row.append(physics.model.id2name(i, 'body'))
      # row.append(physics.model.body_pos[i])
      # row.append(physics.model.body_quat[i])
      # row.append(physics.model.body_ipos[i])
      world_pos = physics.data.xpos[i]
      world_rot = physics.data.xmat[i].reshape(3, 3)
      row.append(world_pos)
      row.append(world_rot)

      (com_pos, com_rot) = calcCOMFrame(physics, world_pos, world_rot)
      row.append(com_pos)
      row.append(com_rot)

      (parent_pos, parent_rot) = calcParentFrame(physics, world_pos, world_rot, i)
      row.append(parent_pos)
      row.append(parent_rot)

      writer.writerow(row)

      bodies[physics.model.id2name(i, 'body')] = [world_pos, world_rot, com_pos, com_rot, parent_pos, parent_rot]

def createbodiescsv():
  with open('bodies.csv', 'a') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerow(['frame index', 'body index', 'body name', 'World Position', 'World Orientation', 'COM Position', 'COM Orientation',
      'Parent Position', 'Parent Orientation'])

def calcCOMFrame(physics, pos, rot):
  #assume world frame: (0, 0, 0) origin with (1, 0, 0), (0, 1, 0), (1, 0, 0) frame

  com_pos = physics.named.data.subtree_com['thorax']
  com_rot = physics.named.data.xmat['thorax'].reshape(3, 3)

  new_pos = np.matmul((pos - com_pos), com_rot)
  new_rot = np.matmul(rot.transpose(), com_rot)

  return (new_pos, new_rot)

def calcParentFrame(physics, pos, rot, index):
  parent_id = physics.model.body_parentid[index]
  parent_pos = physics.data.xpos[parent_id]
  parent_rot = physics.data.xmat[parent_id].reshape(3, 3)

  new_pos = np.matmul((pos - parent_pos), parent_rot)
  new_rot = np.matmul(rot.transpose(), parent_rot)

  return (new_pos, new_rot)

def addnoise(data):
  y = []
  for i in range(len(data)):
    y.append(data[i] * np.random.uniform(0.8, 1.2)) # 20% variance
  y = np.array(y)
  x = np.array(range(len(data)))

  yhat = savgol_filter(y, 21, 10) # window size 51, polynomial order 10

  # y = np.array(y).reshape(-1, 1)
  # x = np.array(range(len(data))).reshape(-1, 1)
  # kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
  # gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
  # gp.fit(x, data)

  # yhat, sigma = gp.predict(y, return_std=True)

  ## plot
  
  plt.plot(x,y, label='raw noise')
  plt.plot(x,yhat, color='red', label='filtered noise')
  plt.plot(x, data, color='green', label='original motion')
  plt.legend()
  plt.show()

  return y

def parsedata(filename, sim, noise):
  output_filename = os.path.basename(filename).split('.')[0] + "_joints.csv"
  
  try:
    os.remove("cpp_jnt_pos")
  except OSError:
      pass
  try:
    os.remove("bodies.csv")
  except OSError:
      pass
    

  env = humanoid_CMU.stand()

  # Parse and convert specified clip.
  converted = parse_amc.convert(filename,
                                env.physics, env.control_timestep())

  max_frame = min(FLAGS.max_num_frames, converted.qpos.shape[1] - 1)

  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  createjointcsv(output_filename)
  createbodiescsv()

  ### add noise to data
  if noise:
    data = []
    for i in range(max_frame):
      p_i = converted.qpos[:, i]
      #with env.physics.reset_context():
      data.append(p_i[17])
    newdata = addnoise(np.array(data))
    
  for i in range(int(max_frame)):
    p_i = converted.qpos[:, i]
    with env.physics.reset_context():
      for j in range(len(p_i)):
        if noise:
          ## TODO: add noise
          env.physics.data.qpos[j] = p_i[j]
        else:
          #env.physics.data.qpos[j] = p_i[j]
          if j == 56:
            env.physics.data.qpos[j] = converted.qpos[:, i][j]

      env.physics.step()
      outputbodies(env.physics, i)
      outputjoints2csv(output_filename, i, env.physics)
      outputcppcsv(env.physics)

    if sim:
      video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

    #outputstate2csv(filename + "_state.csv", converted)

  if sim:
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
    for filename in glob.iglob(FLAGS.dir + '/**/*.amc', recursive=True):
      parsedata(filename, FLAGS.simulate, FLAGS.noise)
  elif FLAGS.filename:
    parsedata(FLAGS.filename, FLAGS.simulate, FLAGS.noise)
  else:
    print("Please provide input source.")

if __name__ == '__main__':
  #flags.mark_flag_as_required('filename')
  app.run(main)
