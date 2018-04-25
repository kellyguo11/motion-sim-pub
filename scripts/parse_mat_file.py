import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

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

window = int(len(acc_x) / 22)
ax = []
ay = []
az = []
for i in range(0, len(acc_x), window):
	ax.append(acc_x[i])
	ay.append(acc_y[i])
	az.append(acc_z[i])

x = np.array(range(len(ax)))

plt.figure(1)
plt.title('acceleration')
plt.plot(x, ax, label='acc_x', color='red')
plt.plot(x, ay, label='acc_y', color='green')
plt.plot(x, az, label='acc_z', color='blue')
plt.legend()
plt.show()