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

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'amc file to be converted.')
flags.DEFINE_integer('max_num_frames', 90,
                     'Maximum number of frames for plotting/playback')
flags.DEFINE_string('output_state_csv', None, 'csv file to contain output of state.')
flags.DEFINE_string('output_joints_csv', None, 'csv file to contain output of joints.')

JOINTS_ORDER = ["lfemurrz", "lfemurry", "lfemurrx", "ltibiarx", "lfootrz", "lfootrx", "ltoesrx" ,
                "rfemurrz", "rfemurry", "rfemurrx", "rtibiarx", "rfootrz" , "rfootrx" , "rtoesrx" ,
                "lowerbackrz", "lowerbackry", "lowerbackrx", "upperbackrz", "upperbackry", "upperbackrx", "thoraxrz",
                "thoraxry", "thoraxrx", "lowerneckrz", "lowerneckry", "lowerneckrx", "upperneckrz", "upperneckry",
                "upperneckrx", "headrz", "headry", "headrx", "lclaviclerz", "lclaviclery", "lhumerusrz", 
                "lhumerusry", "lhumerusrx", "lradiusrx", "lwristry", "lhandrz" , "lhandrx" , "lfingersrx",
                "lthumbrz", "lthumbrx", "rclaviclerz", "rclaviclery", "rhumerusrz", "rhumerusry", "rhumerusrx", 
                "rradiusrx", "rwristry", "rhandrz" , "rhandrx" , "rfingersrx", "rthumbrz", "rthumbrx"]

def outputstate2csv(converted):
  if FLAGS.output_state_csv:
    with open(FLAGS.output_state_csv, 'w') as output:
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

def outputjoints2csv(frame, physics):
  # remove free joint 'root'
  joint_angle = physics.position()[7:]
  joint_vel = physics.velocity()[6:]
  jnt_axis = physics.model.jnt_axis[1:]
  joint_axis = physics.data.xaxis[1:]

  if FLAGS.output_joints_csv:
    with open(FLAGS.output_joints_csv, 'a') as output:
      writer = csv.writer(output, delimiter=',')
      for i in range(len(JOINTS_ORDER)):
        joint_name = physics.model.name2id(JOINTS_ORDER[i], 'joint')
        row = [str(frame), joint_name, JOINTS_ORDER[i]]
        
        row.append(joint_angle[i])
        row.append(joint_vel[i])
        row.append(jnt_axis[i])
        row.append(joint_axis[i])

        writer.writerow(row)

def createjointcsv():
  if FLAGS.output_joints_csv:
    with open(FLAGS.output_joints_csv, 'w') as output:
      writer = csv.writer(output, delimiter=',')
      writer.writerow(['frame index', 'joint index', 'joint name', 'joint angle', 'joint velocity', 'local joint axis', 'global joint axis'])

def addnoise(data):
  y = data + np.random.uniform(-100, 100) * 0.2 # 20% variance
  yhat = savgol_filter(y, 51, 10) # window size 51, polynomial order 10

  x = range(56)
  plt.plot(x,y)
  plt.plot(x,yhat, color='red')
  plt.show()

def main(unused_argv):
  env = humanoid_CMU.stand()

  # Parse and convert specified clip.
  converted = parse_amc.convert(FLAGS.filename,
                                env.physics, env.control_timestep())

  max_frame = min(FLAGS.max_num_frames, converted.qpos.shape[1] - 1)

  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  createjointcsv()

  data = []

  for i in range(max_frame):
    p_i = converted.qpos[:, i]
    with env.physics.reset_context():
      for j in range(len(p_i)):
        # relative to body frame
        env.physics.data.qpos[j] = p_i[j]
      outputjoints2csv(i, env.physics)
      data.append(env.physics.data.qpos[i][17])
  addnoise(data)
    
  for i in range(max_frame):
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

    outputstate2csv(converted)

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


if __name__ == '__main__':
  flags.mark_flag_as_required('filename')
  app.run(main)
