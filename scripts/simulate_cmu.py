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
import os, sys
import copy

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'amc file to be converted.')
flags.DEFINE_integer('max_num_frames', 90,
                     'Maximum number of frames for plotting/playback')
#flags.DEFINE_string('output_state_csv', None, 'csv file to contain output of state.')
#flags.DEFINE_string('output_joints_csv', None, 'csv file to contain output of joints.')
flags.DEFINE_string('dir', None, 'directory containing all amc files.')
flags.DEFINE_string('filter', None, 'tsv file containing filtered amc file names.')
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

# bodies that joints are attached to (in order of above)
JOINT_BODIES = ["lfemur", "lfemur", "lfemur", "ltibia", "lfoot", "lfoot", "ltoes",
                "rfemur", "rfemur", "rfemur", "rtibia", "rfoot", "rfoot", "rtoes",
                "lowerback", "lowerback", "lowerback", "upperback", "upperback", "upperback", "thorax",
                "thorax", "thorax", "lowerneck", "lowerneck", "lowerneck", "upperneck", "upperneck",
                "upperneck", "head", "head", "head", "lclavicle", "lclavicle", "lhumerus", 
                "lhumerus", "lhumerus", "lradius", "lwrist", "lhand", "lhand", "lfingers",
                "lthumb", "lthumb", "rclavicle", "rclavicle", "rclavicle", "rhumerus", "rhumerus",
                "rhumerus", "rradius", "rwrist", "rhand", "rhand", "rfingers", "rthumb", "rthumb"]

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

# output all joints information to csv - human readable
def outputjoints2csv(output_joints_csv, frame, physics, bodies, joints):
  joint_angle = physics.position()
  joint_vel = physics.velocity()
  joint_axis = physics.data.xaxis

  if output_joints_csv:
    with open("csv.csv", 'a') as output:
      writer = csv.writer(output, delimiter=',')

      #output freejoint info
      row1 = [str(frame), 0, 'root', [joint_angle[0:3].tolist(), joint_angle[3:7].tolist()], '', [joint_vel[0:3].tolist(), joint_vel[3:6].tolist()], 
        joint_axis[0], bodies['root'][0], bodies['root'][1], bodies['root'][2], bodies['root'][3], bodies['root'][4], bodies['root'][5]]
      writer.writerow(row1)

      # remove freejoint 'root' due to different dimensions
      joint_angle = physics.position()[7:]
      joint_vel = physics.velocity()[6:]
      joint_axis = physics.data.xaxis[1:]

      for i in range(len(JOINTS_ORDER)):
        joint_name = physics.model.name2id(JOINTS_ORDER[i], 'joint')
        row =  [str(frame), joint_name, JOINTS_ORDER[i]]
        
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

        if frame == 0:
          joints[i] = []
        joints[i].append(copy.deepcopy(row))

        writer.writerow(row)

  return joints

# feature packed formatting
def buildFeatures(frame, physics, bodies, joints):
  frame_data_world = []
  frame_data_com = []
  frame_data_parent = []

  joint_angle = physics.position()
  joint_vel = physics.velocity()
  joint_accel = physics.data.qacc
  joint_axis = physics.data.xaxis

  #output freejoint info
  #pos (3), orientation (9), linear velocity (3), angular velocity (3), linear acceleration (3), angular acceleration (3)
  freejoint_data_world = [bodies['root'][0].tolist(), bodies['root'][1].tolist(), joint_vel[0:3].tolist(), joint_vel[3:6].tolist(), joint_accel[0:3].tolist(), joint_accel[3:6].tolist()]
  frame_data_world.append(freejoint_data_world)

  freejoint_data_com = [bodies['root'][2].tolist(), bodies['root'][3].tolist(), joint_vel[0:3].tolist(), joint_vel[3:6].tolist(), joint_accel[0:3].tolist(), joint_accel[3:6].tolist()]
  frame_data_com.append(freejoint_data_com)

  freejoint_data_parent = [bodies['root'][4].tolist(), bodies['root'][5].tolist(), joint_vel[0:3].tolist(), joint_vel[3:6].tolist(), joint_accel[0:3].tolist(), joint_accel[3:6].tolist()]
  frame_data_parent.append(freejoint_data_parent)

  # remove freejoint 'root' due to different dimensions
  joint_angle = physics.position()[7:]
  joint_vel = physics.velocity()[6:]
  joint_accel = physics.data.qacc[6:]
  #joint_axis = physics.data.xaxis[1:]
  joint_axis = physics.model.jnt_axis[1:]

  for i in range(len(JOINTS_ORDER)):
    # pos (3), orientation (3 x 3), axis (3), angle (1), velocity (1), acceleration (1)
    data_world = [bodies[JOINT_BODIES[i]][0].tolist(), bodies[JOINT_BODIES[i]][1].tolist(), joint_axis[i].tolist(), joint_angle[i], joint_vel[i], joint_accel[i]]
    frame_data_world.append(data_world)

    #convertCOMAxis(physics, joint_axis[i]).tolist()
    data_com = [bodies[JOINT_BODIES[i]][2].tolist(), bodies[JOINT_BODIES[i]][3].tolist(), joint_axis[i].tolist(), joint_angle[i], joint_vel[i], joint_accel[i]]
    frame_data_com.append(data_com)

    #convertParentAxis(physics, joint_axis[i], i).tolist()
    data_parent = [bodies[JOINT_BODIES[i]][4].tolist(), bodies[JOINT_BODIES[i]][5].tolist(), joint_axis[i].tolist(), joint_angle[i], joint_vel[i], joint_accel[i]]
    frame_data_parent.append(data_parent)

    # create joints dictionary
    joint_name = physics.model.name2id(JOINTS_ORDER[i], 'joint')
    row =  [str(frame), joint_name, JOINTS_ORDER[i]]
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

    if frame == 0:
      joints[i] = []
    joints[i].append(copy.deepcopy(row))

  return joints, frame_data_world, frame_data_com, frame_data_parent

# initializes joints output csv with headers
def createjointcsv(output_joints_csv):
  with open(output_joints_csv, 'w') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerow(['frame index', 'joint index', 'joint name', 'joint angle/position', 'joint angle (deg)', 'joint velocity', 
      'cartesian joint axis', 'World Position', 'World Orientation', 'COM Position', 'COM Orientation',
      'Parent Position', 'Parent Orientation'])

def outputcppcsv(filename, physics):
  qpos = physics.data.qpos

  with open(filename, 'a') as output:
    writer = csv.writer(output, delimiter=',')
    for i in range(len(qpos)):
      writer.writerow([qpos[i]])

def calcBodyFrames(physics):
  bodies = dict()
  for i in range(physics.model.nbody):
    world_pos = physics.data.xpos[i]
    world_rot = physics.data.xmat[i].reshape(3, 3)

    (com_pos, com_rot) = calcCOMFrame(physics, world_pos, world_rot)
    (parent_pos, parent_rot) = calcParentFrame(physics, world_pos, world_rot, i)

    bodies[physics.model.id2name(i, 'body')] = [world_pos, world_rot, com_pos, com_rot, parent_pos, parent_rot]
  return bodies

def outputbodies(filename, physics, frame):
  calcBodyFrames(physics)
  with open(filename, 'a') as output:
    writer = csv.writer(output, delimiter=',')
    for i in range(physics.model.nbody):
      body_name = physics.model.id2name(i, 'body')
      row = [frame]
      row.append(i)
      row.append(body_name)
      # world frame
      row.append(bodies[body_name][0])
      row.append(bodies[body_name][1])
      # COM frame
      row.append(bodies[body_name][2])
      row.append(bodies[body_name][3])
      #parent frame
      row.append(bodies[body_name][4])
      row.append(bodies[body_name][5])

      writer.writerow(row)

def createbodiescsv(filename):
  with open(filename, 'a') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerow(['frame index', 'body index', 'body name', 'World Position', 'World Orientation', 'COM Position', 'COM Orientation',
      'Parent Position', 'Parent Orientation'])

def convertCOMAxis(physics, axis):
  com_pos = physics.named.data.subtree_com['thorax']
  com_rot = physics.named.data.xmat['thorax'].reshape(3, 3)

  new_axis = np.matmul((axis - com_pos), com_rot)
  return new_axis

def convertParentAxis(physics, axis, i):
  index = physics.model.name2id(JOINT_BODIES[i], 'body')
  parent_id = physics.model.body_parentid[index]
  parent_pos = physics.data.xpos[parent_id]
  parent_rot = physics.data.xmat[parent_id].reshape(3, 3)

  new_axis = np.matmul((axis - parent_pos), parent_rot)
  return new_axis

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

def visualizeJoint(index, joints):
  data = joints[index]
  world_x = []
  world_y = []
  world_z = []

  com_x = []
  com_y = []
  com_z = []

  par_x = []
  par_y = []
  par_z = []

  x = np.array(range(len(data)))

  for i in range(len(data)):
    world_x.append(data[i][7][0])
    world_y.append(data[i][7][1])
    world_z.append(data[i][7][2])

    com_x.append(data[i][9][0])
    com_y.append(data[i][9][1])
    com_z.append(data[i][9][2])

    par_x.append(data[i][11][0])
    par_y.append(data[i][11][1])
    par_z.append(data[i][11][2])

  plt.figure(1)

  plt.subplot(221)
  plt.title('world')
  plt.plot(x, world_x, label='world_x', color='blue')
  plt.plot(x, world_y, label='world_y', color='red')
  plt.plot(x, world_z, label='world_z', color='green')
  plt.legend()

  plt.subplot(222)
  plt.title('center of mass')
  plt.plot(x, com_x, label='com_x', color='blue')
  plt.plot(x, com_y, label='com_y', color='red')
  plt.plot(x, com_z, label='com_z', color='green')
  plt.legend()

  plt.subplot(223)
  plt.title('parent body')
  plt.plot(x, par_x, label='par_x', color='blue')
  plt.plot(x, par_y, label='par_y', color='red')
  plt.plot(x, par_z, label='par_z', color='green')
  plt.legend()

  plt.show(block=False)

def cleanFiles(prefix):
  try:
    os.remove(prefix + "_bodies.csv")
  except OSError:
      pass
  try:
    os.remove(prefix + "_cpp")
  except OSError:
      pass

  try:
    os.remove(prefix + "_features.txt")
  except OSError:
      pass

def writeOutput(filename, output):
  print("Writing to file: " + filename)
  file = open(filename, 'w')
  for row in output:
    file.write("%s\n" % row)

def parsedata(filename, sim, noise):
  file_prefix = 'features/' + os.path.basename(filename).split('.')[0]
  cleanFiles('features/' + os.path.basename(filename).split('.')[0])

  env = humanoid_CMU.stand()

  # Parse and convert specified clip.
  converted = parse_amc.convert(filename,
                                env.physics, env.control_timestep())

  frame_num = converted.qpos.shape[1] - 1
  max_frame = min(FLAGS.max_num_frames, converted.qpos.shape[1] - 1)

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
    
  for i in range(int(max_frame)):
    p_i = converted.qpos[:, i]
    with env.physics.reset_context():
      for j in range(len(p_i)):
        if noise:
          ## TODO: add noise
          env.physics.data.qpos[j] = p_i[j]
        else:
          env.physics.data.qpos[j] = p_i[j]
          if j == 56:
            env.physics.data.qpos[j] = converted.qpos[:, i][j]

      env.physics.step()
      print("Processing frame " + str(i))
      #outputbodies(os.path.basename(filename).split('.')[0] + "_bodies.csv", env.physics, i)
      bodies = calcBodyFrames(env.physics)
      #joints = outputjoints2csv(output_filename, i, env.physics, bodies, joints)
      joints, frame_data_world, frame_data_com, frame_data_parent = buildFeatures(i, env.physics, bodies, joints)
      output_world.append(frame_data_world)
      output_com.append(frame_data_com)
      output_parent.append(frame_data_parent)
      #outputcppcsv(os.path.basename(filename).split('.')[0] + "_cpp", env.physics)

    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

    #outputstate2csv(filename + "_state.csv", converted)
  writeOutput(file_prefix + "_features_world.txt", output_world)
  writeOutput(file_prefix + "_features_com.txt", output_com)
  writeOutput(file_prefix + "_features_parent.txt", output_parent)

  if sim:
    visualizeJoint(49, joints)

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

def filterdata(filename, sim, noise):
  folder = os.path.join(os.path.realpath('..'), 'motion-sim/cmu-data/all_asfamc/subjects/')
  with open(filename,'r') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader, None)  # skip header
    for row in reader:
      amc_file_name = row[0]
      subj_num = amc_file_name.split('_')[0]
      amc_file_path = folder + subj_num + '/' + amc_file_name
      print(amc_file_path)
      parsedata(amc_file_path, sim, noise)

def main(unused_argv):
  if FLAGS.dir:
    for filename in glob.iglob(FLAGS.dir + '/**/*.amc', recursive=True):
      #try:
      parsedata(filename, FLAGS.simulate, FLAGS.noise)
      #except Exception as e:
        #print(e)
  elif FLAGS.filename:
    #try:
    parsedata(FLAGS.filename, FLAGS.simulate, FLAGS.noise)
    #except Exception as e:
    #  print(e)
  elif FLAGS.filter:
    filterdata(FLAGS.filter, FLAGS.simulate, FLAGS.noise)
  else:
    print("Please provide input source.")

if __name__ == '__main__':
  #flags.mark_flag_as_required('filename')
  app.run(main)
