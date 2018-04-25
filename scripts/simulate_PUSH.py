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
import glob
import os, sys
import copy
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'txt file to be converted.')
flags.DEFINE_string('dir', None, 'directory containing all txt files.')
flags.DEFINE_boolean('simulate', False, 'Simulate results.')
flags.DEFINE_integer('maxframe', 1000, 'Max frames to simulate.')

JOINTS_ORDER = ["lfemurrz", "lfemurry", "lfemurrx", "ltibiarx", "lfootrz", "lfootrx", "ltoesrx" ,
                "rfemurrz", "rfemurry", "rfemurrx", "rtibiarx", "rfootrz" , "rfootrx" , "rtoesrx" ,
                "lowerbackrz", "lowerbackry", "lowerbackrx", "upperbackrz", "upperbackry", "upperbackrx", "thoraxrz",
                "thoraxry", "thoraxrx", "lowerneckrz", "lowerneckry", "lowerneckrx", "upperneckrz", "upperneckry",
                "upperneckrx", "headrz", "headry", "headrx", "lclaviclerz", "lclaviclery", "lhumerusrz", 
                "lhumerusry", "lhumerusrx", "lradiusrx", "lwristry", "lhandrz" , "lhandrx" , "lfingersrx",
                "lthumbrz", "lthumbrx", "rclaviclerz", "rclaviclery", "rhumerusrz", "rhumerusry", "rhumerusrx", 
                "rradiusrx", "rwristry", "rhandrz" , "rhandrx" , "rfingersrx", "rthumbrz", "rthumbrx"]

INIT_QPOS = [0, 15, 0, 0, 0, -15, 0, 0]

JOINT_MAPPING = dict()
JOINT_MAPPING[0] = "rfemurrx" #x, y, z
JOINT_MAPPING[1] = "rfemurrz" #x, y, z
JOINT_MAPPING[2] = "rtibiarx"
JOINT_MAPPING[3] = "rfootrx" # also rz
JOINT_MAPPING[4] = "lfemurrx" #x, y, z
JOINT_MAPPING[5] = "lfemurrz" #x, y, z
JOINT_MAPPING[6] = "ltibiarx"
JOINT_MAPPING[7] = "lfootrx" # also rz


def parseData(filename, sim = True, maxframe = 1000):
  file = np.load(filename)

  # set up mujoco simulation
  env = humanoid_CMU.stand()
  num_frame = len(file)
  max_frame = np.minimum(maxframe, num_frame)
  width = 480
  height = 480
  if sim:
    video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  joint_positions = []
  rtibia_body_index = env.physics.model.name2id('rtibia', 'body')
  rfoot_body_index = env.physics.model.name2id('rfoot', 'body')
  ltibia_body_index = env.physics.model.name2id('ltibia', 'body')
  lfoot_body_index = env.physics.model.name2id('lfoot', 'body')

  for i in tqdm(range(int(max_frame))):
    with env.physics.reset_context():
      for j in range(len(file[i])):
        joint_name = JOINT_MAPPING[j]
        joint_index = env.physics.model.name2id(joint_name, 'joint')
        if j == 0 or j == 4:
          env.physics.data.qpos[joint_index + 6] = -np.radians(file[i][j]) + np.radians(INIT_QPOS[j])
        else:
          env.physics.data.qpos[joint_index + 6] = np.radians(file[i][j]) + np.radians(INIT_QPOS[j])

      env.physics.step()
      ##### output body/joint positions #####
      pos = [env.physics.data.xpos[lfoot_body_index], env.physics.data.xpos[rfoot_body_index], 
        env.physics.data.xpos[ltibia_body_index], env.physics.data.xpos[rtibia_body_index]]
      joint_positions.append(copy.deepcopy(pos))

    if sim:
      video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

  outputPositions(filename, joint_positions)

  if sim:
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

def outputPositions(filename, pos):
  filename = filename + "_positions.npy"
  np.save(filename, pos)

def main(unused_argv):
  if FLAGS.dir:
    for filename in glob.iglob(FLAGS.dir):
      parseData(filename, False, FLAGS.maxframe)
  elif FLAGS.filename:
    parseData(FLAGS.filename, FLAGS.simulate, FLAGS.maxframe)
  else:
    print("Please provide input source.")

if __name__ == '__main__':
  #flags.mark_flag_as_required('filename')
  app.run(main)
