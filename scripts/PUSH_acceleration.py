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

import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import os, sys
import scipy.io as sio
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn import gaussian_process

import transforms3d
import rmsd

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'txt file to be converted.')
flags.DEFINE_string('dir', None, 'directory containing all txt files.')

height = 0
ms2_to_G = 0.101972
num_frames = 0

pos_x = dict()
pos_y = dict()
pos_z = dict()

DELTA_T = 1/30

def find_rotation_matrix(ax, ay, az, imu_ax, imu_ay, imu_az):
  print("Finding rotation matrix...")
  rx = 0
  ry = 0
  rz = 0

  diff_a = []
  imu_a = []

  # Kabsch algorithm setup
  for i in range(np.minimum(len(ax), len(imu_ax))):
    diff_a.append([ax[i], ay[i], az[i]])
  for i in range(np.minimum(len(ax), len(imu_ax))):
    imu_a.append([imu_ax[i], imu_ay[i], imu_az[i] + 20])

  diff_a = np.array(diff_a)
  imu_a = np.array(imu_a)

  print("Rotation matrix: ", rmsd.kabsch(diff_a, imu_a))
  print("RMSD: ", rmsd.kabsch_rmsd(diff_a, imu_a))
  # diff_a -= rmsd.centroid(diff_a)
  # imu_a -= rmsd.centroid(imu_a)
  # print("Rotation matrix after translation: ", rmsd.kabsch(diff_a, imu_a))
  # print("RMSD after translation: ", rmsd.kabsch_rmsd(diff_a, imu_a))

  rotated_a = rmsd.kabsch_rotate(diff_a, imu_a)
  r_ax = []
  r_ay = []
  r_az = []
  for i in range(len(rotated_a)):
    r_ax.append(rotated_a[i][0])
    r_ay.append(rotated_a[i][1])
    r_az.append(rotated_a[i][2])

  x_axis = np.array(range(len(r_ax)))
  plt.figure(5)
  plt.title("Rotated Differentiated Acceleration")
  plt.plot(x_axis, r_ax, label='x', color='red')
  plt.plot(x_axis, r_ay, label='y', color='green')
  plt.plot(x_axis, r_az, label='z', color='blue')
  plt.legend()

def parse_IMU(filename):
  print("Parsing IMU file...")

  lshank_x = []
  lshank_y = []
  lshank_z = []
  lthigh_x = []
  lthigh_y = []
  lthigh_z = []
  rshank_x = []
  rshank_y = []
  rshank_z = []
  rthigh_x = []
  rthigh_y = []
  rthigh_z = []

  filename = filename.replace('IK_cut_spline', 'IMU_cut_spline')
  data = np.load(filename)

  for row in data:
    lshank_x.append(row[20])
    lshank_y.append(row[21])
    lshank_z.append(row[22])
    lthigh_x.append(row[46])
    lthigh_y.append(row[47])
    lthigh_z.append(row[48])
    rshank_x.append(row[33])
    rshank_y.append(row[34])
    rshank_z.append(row[35])
    rthigh_x.append(row[59])
    rthigh_y.append(row[60])
    rthigh_z.append(row[61])

  # lshank
  x_axis = np.array(range(len(lshank_x)))
  plt.figure(5)
  plt.subplot(211)
  plt.title("IMU Acceleration -- LShank")
  plt.plot(x_axis, lshank_x, label='x', color='red')
  plt.plot(x_axis, lshank_y, label='y', color='green')
  plt.plot(x_axis, lshank_z, label='z', color='blue')
  plt.legend()

  sav_x, sav_y, sav_z = savgol(lshank_x, lshank_y, lshank_z, 151)
  x_axis = np.array(range(len(sav_x)))
  plt.subplot(212)
  plt.title("IMU Acceleration with Savgol Filter -- LShank")
  plt.plot(x_axis, sav_x, label='x', color='red')
  plt.plot(x_axis, sav_y, label='y', color='green')
  plt.plot(x_axis, sav_z, label='z', color='blue')
  plt.legend()

  # lthigh
  x_axis = np.array(range(len(lthigh_x)))
  plt.figure(6)
  plt.subplot(211)
  plt.title("IMU Acceleration -- LThigh")
  plt.plot(x_axis, lthigh_x, label='x', color='red')
  plt.plot(x_axis, lthigh_y, label='y', color='green')
  plt.plot(x_axis, lthigh_z, label='z', color='blue')
  plt.legend()

  sav_x, sav_y, sav_z = savgol(lthigh_x, lthigh_y, lthigh_z, 151)
  x_axis = np.array(range(len(sav_x)))
  plt.subplot(212)
  plt.title("IMU Acceleration with Savgol Filter -- LThigh")
  plt.plot(x_axis, sav_x, label='x', color='red')
  plt.plot(x_axis, sav_y, label='y', color='green')
  plt.plot(x_axis, sav_z, label='z', color='blue')
  plt.legend()

  # rshank
  x_axis = np.array(range(len(rshank_x)))
  plt.figure(7)
  plt.subplot(211)
  plt.title("IMU Acceleration -- RShank")
  plt.plot(x_axis, rshank_x, label='x', color='red')
  plt.plot(x_axis, rshank_y, label='y', color='green')
  plt.plot(x_axis, rshank_z, label='z', color='blue')
  plt.legend()

  sav_x, sav_y, sav_z = savgol(rshank_x, rshank_y, rshank_z, 151)
  x_axis = np.array(range(len(sav_x)))
  plt.subplot(212)
  plt.title("IMU Acceleration with Savgol Filter -- RShank")
  plt.plot(x_axis, sav_x, label='x', color='red')
  plt.plot(x_axis, sav_y, label='y', color='green')
  plt.plot(x_axis, sav_z, label='z', color='blue')
  plt.legend()

  # rthigh
  x_axis = np.array(range(len(rthigh_x)))
  plt.figure(8)
  plt.subplot(211)
  plt.title("IMU Acceleration -- RThigh")
  plt.plot(x_axis, rthigh_x, label='x', color='red')
  plt.plot(x_axis, rthigh_y, label='y', color='green')
  plt.plot(x_axis, rthigh_z, label='z', color='blue')
  plt.legend()

  sav_x, sav_y, sav_z = savgol(rthigh_x, rthigh_y, rthigh_z, 151)
  x_axis = np.array(range(len(sav_x)))
  plt.subplot(212)
  plt.title("IMU Acceleration with Savgol Filter -- RThigh")
  plt.plot(x_axis, sav_x, label='x', color='red')
  plt.plot(x_axis, sav_y, label='y', color='green')
  plt.plot(x_axis, sav_z, label='z', color='blue')
  plt.legend()

  return [sav_x, sav_y, sav_z]

def savgol(pos_x, pos_y, pos_z, window):
  x = np.array(range(len(pos_x)))
  sav_x = savgol_filter(pos_x, window, 3)
  sav_y = savgol_filter(pos_y, window, 3)
  sav_z = savgol_filter(pos_z, window, 3)
  return [sav_x, sav_y, sav_z]

def calc_velocity(pos_x, pos_y, pos_z):
  vel_x = []
  vel_y = []
  vel_z = []

  for i in range(len(pos_x) - 1):
    delta_x = pos_x[i + 1] - pos_x[i]
    delta_y = pos_y[i + 1] - pos_y[i]
    delta_z = pos_z[i + 1] - pos_z[i]

    vel_x.append(delta_x / DELTA_T)
    vel_y.append(delta_y / DELTA_T)
    vel_z.append(delta_z / DELTA_T)

  return [vel_x, vel_y, vel_z]

def calc_acceleration(vel_x, vel_y, vel_z):
  acc_x = []
  acc_y = []
  acc_z = []

  for i in range(len(vel_x) - 1):
    delta_x = vel_x[i + 1] - vel_x[i]
    delta_y = vel_y[i + 1] - vel_y[i]
    delta_z = vel_z[i + 1] - vel_z[i]

    acc_x.append(delta_x / DELTA_T)
    acc_y.append(delta_y / DELTA_T )
    acc_z.append(delta_z / DELTA_T )

  return [acc_x, acc_y, acc_z]

def parse_data(filename):
  print("Parsing joint data...")
  global num_frames, pos_x, pos_y, pos_z

  filename = filename + "_positions.npy"
  file = np.load(filename)

  lshank_x = []
  lshank_y = []
  lshank_z = []
  lthigh_x = []
  lthigh_y = []
  lthigh_z = []
  rshank_x = []
  rshank_y = []
  rshank_z = []
  rthigh_x = []
  rthigh_y = []
  rthigh_z = []

  for row in file:
    lshank_x.append(row[0][0])
    lshank_y.append(row[0][1])
    lshank_z.append(row[0][2])
    rshank_x.append(row[1][0])
    rshank_y.append(row[1][1])
    rshank_z.append(row[1][2])
    lthigh_x.append(row[2][0])
    lthigh_y.append(row[2][1])
    lthigh_z.append(row[2][2])
    rthigh_x.append(row[3][0])
    rthigh_y.append(row[3][1])
    rthigh_z.append(row[3][2])

  # lshank
  plt.figure(1)
  plot_acc(lshank_x, lshank_y, lshank_z, 321, "Raw Position -- LShank")
  sav_x, sav_y, sav_z = savgol(lshank_x, lshank_y, lshank_z, 151)
  plot_acc(sav_x, sav_y, sav_z, 322, "Savgol Filter Position -- LShank")

  vel_x, vel_y, vel_z = calc_velocity(lshank_x, lshank_y, lshank_z)
  plot_acc(vel_x, vel_y, vel_z, 323, "Differentiated Velocity -- LShank")
  sav_vel_x, sav_vel_y, sav_vel_z = calc_velocity(sav_x, sav_y, sav_z)
  sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z = savgol(sav_vel_x, sav_vel_y, sav_vel_z, 151)
  plot_acc(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z, 324, "Savgol Filter^2 Velocity -- LShank")

  acc_x, acc_y, acc_z = calc_acceleration(vel_x, vel_y, vel_z)
  plot_acc(acc_x, acc_y, acc_z, 325, "Differentiated Acceleration -- LShank")
  sav_acc_x, sav_acc_y, sav_acc_z = calc_acceleration(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z)
  plot_acc(sav_acc_x, sav_acc_y, sav_acc_z, 326, "Savgol Filter^2 Acceleration -- LShank")

  # lthigh
  plt.figure(2)
  plot_acc(lthigh_x, lthigh_y, lthigh_z, 321, "Raw Position -- LThigh")
  sav_x, sav_y, sav_z = savgol(lthigh_x, lthigh_y, lthigh_z, 151)
  plot_acc(sav_x, sav_y, sav_z, 322, "Savgol Filter Position -- LThigh")

  vel_x, vel_y, vel_z = calc_velocity(lthigh_x, lthigh_y, lthigh_z)
  plot_acc(vel_x, vel_y, vel_z, 323, "Differentiated Velocity -- LThigh")
  sav_vel_x, sav_vel_y, sav_vel_z = calc_velocity(sav_x, sav_y, sav_z)
  sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z = savgol(sav_vel_x, sav_vel_y, sav_vel_z, 151)
  plot_acc(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z, 324, "Savgol Filter^2 Velocity -- LThigh")

  acc_x, acc_y, acc_z = calc_acceleration(vel_x, vel_y, vel_z)
  plot_acc(acc_x, acc_y, acc_z, 325, "Differentiated Acceleration -- LThigh")
  sav_acc_x, sav_acc_y, sav_acc_z = calc_acceleration(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z)
  plot_acc(sav_acc_x, sav_acc_y, sav_acc_z, 326, "Savgol Filter^2 Acceleration -- LThigh")

  # rshank
  plt.figure(3)
  plot_acc(rshank_x, rshank_y, rshank_z, 321, "Raw Position -- RShank")
  sav_x, sav_y, sav_z = savgol(rshank_x, rshank_y, rshank_z, 151)
  plot_acc(sav_x, sav_y, sav_z, 322, "Savgol Filter Position -- RShank")

  vel_x, vel_y, vel_z = calc_velocity(rshank_x, rshank_y, rshank_z)
  plot_acc(vel_x, vel_y, vel_z, 323, "Differentiated Velocity -- RShank")
  sav_vel_x, sav_vel_y, sav_vel_z = calc_velocity(sav_x, sav_y, sav_z)
  sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z = savgol(sav_vel_x, sav_vel_y, sav_vel_z, 151)
  plot_acc(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z, 324, "Savgol Filter^2 Velocity -- RShank")

  acc_x, acc_y, acc_z = calc_acceleration(vel_x, vel_y, vel_z)
  plot_acc(acc_x, acc_y, acc_z, 325, "Differentiated Acceleration -- RShank")
  sav_acc_x, sav_acc_y, sav_acc_z = calc_acceleration(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z)
  plot_acc(sav_acc_x, sav_acc_y, sav_acc_z, 326, "Savgol Filter^2 Acceleration -- RShank")

  # rthigh
  plt.figure(4)
  plot_acc(rthigh_x, rthigh_y, rthigh_z, 321, "Raw Position -- RThigh")
  sav_x, sav_y, sav_z = savgol(rthigh_x, rthigh_y, rthigh_z, 151)
  plot_acc(sav_x, sav_y, sav_z, 322, "Savgol Filter Position -- RThigh")

  vel_x, vel_y, vel_z = calc_velocity(rthigh_x, rthigh_y, rthigh_z)
  plot_acc(vel_x, vel_y, vel_z, 323, "Differentiated Velocity -- RThigh")
  sav_vel_x, sav_vel_y, sav_vel_z = calc_velocity(sav_x, sav_y, sav_z)
  sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z = savgol(sav_vel_x, sav_vel_y, sav_vel_z, 151)
  plot_acc(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z, 324, "Savgol Filter^2 Velocity -- RThigh")

  acc_x, acc_y, acc_z = calc_acceleration(vel_x, vel_y, vel_z)
  plot_acc(acc_x, acc_y, acc_z, 325, "Differentiated Acceleration -- RThigh")
  sav_acc_x, sav_acc_y, sav_acc_z = calc_acceleration(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z)
  plot_acc(sav_acc_x, sav_acc_y, sav_acc_z, 326, "Savgol Filter^2 Acceleration -- RThigh")

  return [sav_acc_x, sav_acc_y, sav_acc_z]

def diff_acc(m_ax, m_ay, m_az, ax, ay, az):
  diff_x = []
  diff_y = []
  diff_z = []

  for i in range(len(ax)):
    diff_x.append(ax[i] - m_ax[i])
    diff_y.append(ay[i] - m_ay[i])
    diff_z.append(az[i] - m_az[i])

  return [diff_x, diff_y, diff_z]

def plot_acc(x, y, z, fig, title):
  x_axis = np.array(range(len(x)))
  plt.subplot(fig)
  plt.title(title)
  plt.plot(x_axis, x, label='x', color='red')
  plt.plot(x_axis, y, label='y', color='green')
  plt.plot(x_axis, z, label='z', color='blue')
  plt.legend()

def main(unused_argv):
  if FLAGS.dir:
    for filename in glob.iglob(FLAGS.dir):
      acc_x, acc_y, acc_z = parse_data(filename)
      plot_acc(acc_x, acc_y, acc_z)
  elif FLAGS.filename:
    acc_x, acc_y, acc_z = parse_data(FLAGS.filename)
    ax, ay, az = parse_IMU(FLAGS.filename)
    find_rotation_matrix(acc_x, acc_y, acc_z, ax, ay, az)
    plt.show()
  else:
    print("Please provide input source.")

if __name__ == '__main__':
  #flags.mark_flag_as_required('filename')
  app.run(main)
