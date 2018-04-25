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

def rotate_vector(ax, ay, az, m_ax, m_ay, m_az):
  window = int(len(m_ax) / 77)
  rotation_matrix_list = []
  for i in range(num_frames - 2):

    error = 100
    e = m_ax[window*i]*0.1
    x_axis_1 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    x_axis_2 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    x_axis_3 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    while error > np.absolute(e):
      x_axis_1 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      x_axis_2 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      x_axis_3 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      new_ax = ((
        (pos_x[10][i + 2]*x_axis_1[0] + pos_x[10][i + 2]*x_axis_1[1] + pos_x[10][i + 2]*x_axis_1[2])
        - (2*(pos_x[10][i + 1]*x_axis_2[0] + pos_x[10][i + 1]*x_axis_2[1] + pos_x[10][i + 1]*x_axis_2[2])) 
        + (pos_x[10][i]*x_axis_3[0] + pos_x[10][i]*x_axis_3[1] + pos_x[10][i]*x_axis_3[2]))
        / (DELTA_T*DELTA_T) / 9.18
        )
      error = np.absolute(new_ax - m_ax[i*window])
      print('x_error: ' + str(error))

    error = 100
    e = m_ay[window*i]*0.1
    y_axis_1 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    y_axis_2 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    y_axis_3 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    while error > np.absolute(e):
      y_axis_1 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      y_axis_2 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      y_axis_3 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      new_ay = ((
        (pos_y[10][i + 2]*y_axis_1[0] + pos_y[10][i + 2]*y_axis_1[1] + pos_y[10][i + 2]*y_axis_1[2])
        - (2*(pos_y[10][i + 1]*y_axis_2[0] + pos_y[10][i + 1]*y_axis_2[1] + pos_y[10][i + 1]*y_axis_2[2])) 
        + (pos_y[10][i]*y_axis_3[0] + pos_y[10][i]*y_axis_3[1] + pos_y[10][i]*y_axis_3[2]))
        / (DELTA_T*DELTA_T) / 9.18
        )
      error = np.absolute(new_ay - m_ay[i*window])
      print('y_error: ' + str(error))

    error = 100
    e = m_az[window*i]*0.1
    z_axis_1 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    z_axis_2 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    z_axis_3 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    while error > np.absolute(e):
      z_axis_1 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      z_axis_2 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      z_axis_3 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
      new_az = ((
        (pos_z[10][i + 2]*z_axis_1[0] + pos_z[10][i + 2]*z_axis_1[1] + pos_z[10][i + 2]*z_axis_1[2])
        - (2*(pos_z[10][i + 1]*z_axis_2[0] + pos_z[10][i + 1]*z_axis_2[1] + pos_z[10][i + 1]*z_axis_2[2])) 
        + (pos_z[10][i]*z_axis_3[0] + pos_z[10][i]*z_axis_3[1] + pos_z[10][i]*z_axis_3[2]))
        / (DELTA_T*DELTA_T) / 9.18
        )
      error = np.absolute(new_az - m_az[i*window])
      print('z_error: ' + str(error))

    rotation_matrix_list.append([[x_axis_1, y_axis_1, z_axis_1], [x_axis_2, y_axis_2, z_axis_2], [x_axis_3, y_axis_3, z_axis_3]])
  return rotation_matrix_list

def find_rotation_matrix(ax, ay, az, imu_ax, imu_ay, imu_az):
  print("Finding rotation matrix...")
  rx = 0
  ry = 0
  rz = 0

  diff_a = []
  imu_a = []

  # Kabsch algorithm setup
  for i in range(np.minimum(len(ax), len(imu_ax))):
    diff_a.append([ax[i]/9.81, ay[i]/9.81, az[i]/9.81])
  for i in range(np.minimum(len(ax), len(imu_ax))):
    imu_a.append([imu_ax[i], imu_ay[i], imu_az[i]])

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

def parse_mat(frames):
  print("Parsing matlab file...")

  acc_x = []
  acc_y = []
  acc_z = []
  gyro_x = []
  gyro_y = []
  gyro_z = []

  data = sio.loadmat("/home/kelly/Documents/motion-sim/UTD-MHAD/a1_s1_t1_inertial.mat")
  i = 0
  for row in data["d_iner"]:
    acc_x.append(row[0])
    acc_y.append(row[1])
    acc_z.append(row[2])
    gyro_x.append(row[3])
    gyro_y.append(row[4])
    gyro_z.append(row[5])
    i += 1

  x_axis = np.array(range(len(acc_x)))
  plt.figure(4)
  plt.subplot(311)
  plt.title("IMU Acceleration")
  plt.plot(x_axis, acc_x, label='x', color='red')
  plt.plot(x_axis, acc_y, label='y', color='green')
  plt.plot(x_axis, acc_z, label='z', color='blue')
  plt.legend()

  np_acc_x = np.array(acc_x)
  np_acc_y = np.array(acc_y)
  np_acc_z = np.array(acc_z)

  ax_interp = interpolate.interp1d(np.arange(np_acc_x.size), np_acc_x)
  ax = ax_interp(np.linspace(0, np_acc_x.size - 1, frames))
  ay_interp = interpolate.interp1d(np.arange(np_acc_y.size), np_acc_y)
  ay = ay_interp(np.linspace(0, np_acc_y.size - 1, frames))
  az_interp = interpolate.interp1d(np.arange(np_acc_z.size), np_acc_z)
  az = az_interp(np.linspace(0, np_acc_z.size - 1, frames))

  x_axis = np.array(range(len(ax)))
  plt.subplot(312)
  plt.title("Interpolated Down-sampled IMU Acceleration")
  plt.plot(x_axis, ax, label='x', color='red')
  plt.plot(x_axis, ay, label='y', color='green')
  plt.plot(x_axis, az, label='z', color='blue')
  plt.legend()

  sav_x, sav_y, sav_z = savgol(ax, ay, az)
  x_axis = np.array(range(len(sav_x)))
  plt.subplot(313)
  plt.title("Down-sampled IMU Acceleration with Savgol Filter")
  plt.plot(x_axis, sav_x, label='x', color='red')
  plt.plot(x_axis, sav_y, label='y', color='green')
  plt.plot(x_axis, sav_z, label='z', color='blue')
  plt.legend()

  return [sav_x, sav_y, sav_z]

def unvariate_spline(pos_x, pos_y, pos_z):
  x = np.array(range(len(pos_x)))
  spl_x = UnivariateSpline(x, pos_x)
  spl_y = UnivariateSpline(x, pos_y)
  spl_z = UnivariateSpline(x, pos_z)

  spl_x_val = []
  spl_y_val = []
  spl_z_val = []

  for i in range(len(pos_x)):
    spl_x_val.append(spl_x(i))
    spl_y_val.append(spl_y(i))
    spl_z_val.append(spl_z(i))

  return [spl_x_val, spl_y_val, spl_z_val]

def polyfit(pos_x, pos_y, pos_z):
  x = np.array(range(len(pos_x)))
  fit_x = np.poly1d(np.polyfit(x, pos_x, 12))
  fit_y = np.poly1d(np.polyfit(x, pos_y, 5))
  fit_z = np.poly1d(np.polyfit(x, pos_z, 6))

  fit_x_val = []
  fit_y_val = []
  fit_z_val = []

  for i in range(len(pos_x)):
    fit_x_val.append(fit_x(i))
    fit_y_val.append(fit_y(i))
    fit_z_val.append(fit_z(i))

  return [fit_x_val, fit_y_val, fit_z_val]

def savgol(pos_x, pos_y, pos_z):
  x = np.array(range(len(pos_x)))
  sav_x = savgol_filter(pos_x, 11, 3)
  sav_y = savgol_filter(pos_y, 11, 3)
  sav_z = savgol_filter(pos_z, 11, 3)
  return [sav_x, sav_y, sav_z]

def moving_average(pos_x, pos_y, pos_z):
  window = np.ones(int(5))/float(5)
  avg_x = np.convolve(pos_x, window, 'valid')
  avg_y = np.convolve(pos_y, window, 'valid')
  avg_z = np.convolve(pos_z, window, 'valid')
  return [avg_x, avg_y, avg_z]

def gaussian(pos_x, pos_y, pos_z):
  x = np.array(range(len(pos_x))).reshape(-1, 1)
  pos_x = np.array(pos_x).reshape(-1, 1)
  pos_y = np.array(pos_y).reshape(-1, 1)
  pos_z = np.array(pos_z).reshape(-1, 1)
  
  kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
  gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
  gp.fit(x, pos_x)
  gauss_x, sigma = gp.predict(pos_x, return_std=True)
  gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
  gp.fit(x, pos_y)
  gauss_y, sigma = gp.predict(pos_y, return_std=True)
  gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
  gp.fit(x, pos_z)
  gauss_z, sigma = gp.predict(pos_z, return_std=True)

  return [gauss_x, gauss_y, gauss_z]

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
  file = open(filename, 'r')
  lines = file.read().split('\n')

  num_frames = 0
  num_joints = 0
  min_z = 10
  max_z = -10

  j = 0
  # split by joints
  for joints in lines:
    if joints != '':
      num_joints += 1
      # split by frames
      frames = joints.split(',')
      if j not in pos_x.keys():
        pos_x[j] = []
        pos_y[j] = []
        pos_z[j] = []
      # add position of each joint per frame
      for f in frames:
        positions = f.split(' ')
        pos_x[j].append(float(positions[0]))
        pos_y[j].append(float(positions[1]))
        pos_z[j].append(float(positions[2]))
        if float(positions[1]) > max_z and j == 3:
          max_z = float(positions[1])
        if float(positions[1]) < min_z and (j == 15 or j == 19):
          min_z = float(positions[1])
        if j == 0:
          num_frames += 1
      j += 1

  plt.figure(1)
  plot_acc(pos_x[10], pos_y[10], pos_z[10], 211, "Raw Position")
  #fit_x, fit_y, fit_z = polyfit(pos_x[10], pos_y[10], pos_z[10])
  #plot_acc(fit_x, fit_y, fit_z, 222, "Polyfit Position")
  # spl_x, spl_y, spl_z = unvariate_spline(pos_x[10], pos_y[10], pos_z[10])
  # plot_acc(spl_x, spl_y, spl_z, 353, "Univariate Spline Position")
  # gauss_x, gauss_y, gauss_z = gaussian(pos_x[10], pos_y[10], pos_z[10])
  # plot_acc(gauss_x, gauss_y, gauss_z, 233, "Gaussian Process Position")
  sav_x, sav_y, sav_z = savgol(pos_x[10], pos_y[10], pos_z[10])
  plot_acc(sav_x, sav_y, sav_z, 212, "Savgol Filter Position")
  #avg_x, avg_y, avg_z = moving_average(pos_x[10], pos_y[10], pos_z[10])
  #plot_acc(avg_x, avg_y, avg_z, 224, "Moving Average Position")

  plt.figure(2)
  vel_x, vel_y, vel_z = calc_velocity(pos_x[10], pos_y[10], pos_z[10])
  plot_acc(vel_x, vel_y, vel_z, 211, "Differentiated Velocity")
  #fit_vel_x, fit_vel_y, fit_vel_z = calc_velocity(fit_x, fit_y, fit_z)
  #plot_acc(fit_vel_x, fit_vel_y, fit_vel_z, 322, "Polyfit Velocity")
  sav_vel_x, sav_vel_y, sav_vel_z = calc_velocity(sav_x, sav_y, sav_z)
  sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z = savgol(sav_vel_x, sav_vel_y, sav_vel_z)
  #plot_acc(sav_vel_x, sav_vel_y, sav_vel_z, 323, "Savgol Filter Velocity")
  plot_acc(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z, 212, "Savgol Filter^2 Velocity")
  #avg_vel_x, avg_vel_y, avg_vel_z = calc_velocity(avg_x, avg_y, avg_z)
  #plot_acc(avg_vel_x, avg_vel_y, avg_vel_z, 325, "Moving Average Velocity")

  plt.figure(3)
  acc_x, acc_y, acc_z = calc_acceleration(vel_x, vel_y, vel_z)
  plot_acc(acc_x, acc_y, acc_z, 211, "Differentiated Acceleration")
  #fit_acc_x, fit_acc_y, fit_acc_z = calc_acceleration(fit_vel_x, fit_vel_y, fit_vel_z)
  #plot_acc(fit_acc_x, fit_acc_y, fit_acc_z, 322, "Polyfit Acceleration")
  #sav_acc_x, sav_acc_y, sav_acc_z = calc_acceleration(sav_vel_x, sav_vel_y, sav_vel_z)
  #plot_acc(sav_acc_x, sav_acc_y, sav_acc_z, 323, "Savgol Filter Acceleration")
  sav_acc_x, sav_acc_y, sav_acc_z = calc_acceleration(sav_sav_vel_x, sav_sav_vel_y, sav_sav_vel_z)
  plot_acc(sav_acc_x, sav_acc_y, sav_acc_z, 212, "Savgol Filter^2 Acceleration")
  #avg_acc_x, avg_acc_y, avg_acc_z = calc_acceleration(avg_vel_x, avg_vel_y, avg_vel_z)
  #plot_acc(avg_acc_x, avg_acc_y, avg_acc_z, 325, "Moving Average Acceleration")

  return [sav_acc_x, sav_acc_y, sav_acc_z, num_frames]

  # new_ax = []
  # new_ay = []
  # new_az = []

    # if i < num_frames - 2:
    #   new_ax.append((pos_x[10][i + 2] - 2*pos_x[10][i + 1] + pos_x[10][i])/(DELTA_T*DELTA_T) / 9.18)
    #   new_ay.append((pos_y[10][i + 2] - 2*pos_y[10][i + 1] + pos_y[10][i])/(DELTA_T*DELTA_T) / 9.18)
    #   new_az.append((pos_z[10][i + 2] - 2*pos_z[10][i + 1] + pos_z[10][i])/(DELTA_T*DELTA_T) / 9.18)

  #plot_acc(vel_x, vel_y, vel_z, 222, "Velocity")

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
    acc_x, acc_y, acc_z, num_frames = parse_data(FLAGS.filename)
    ax, ay, az = parse_mat(num_frames)
    find_rotation_matrix(acc_x, acc_y, acc_z, ax, ay, az)
    plt.show()
  else:
    print("Please provide input source.")

if __name__ == '__main__':
  #flags.mark_flag_as_required('filename')
  app.run(main)
