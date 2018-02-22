import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
from tempfile import NamedTemporaryFile

from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

KEYWORDS = ['basketball', 'bend', 'cartwheel', 'climb', 'dance', 'jump', 'kick', 'punch', 'rotate', 'run', 'sit', 
	'spin', 'stand', 'stretch', 'swing', 'throw', 'turn', 'twist', 'walk']

def getNumFrames(filename):
	env = humanoid_CMU.stand()

  # Parse and convert specified clip.
	converted = parse_amc.convert(filename, env.physics, env.control_timestep())
	frame_num = converted.qpos.shape[1] - 1
	return frame_num

def parseMapping():
	folder_path = 'cmu-data/all_asfamc/subjects/'

	with open('cmu_desc.tsv','r') as file, open('cmu_mocap_labels.tsv', 'w') as out:
		reader = csv.reader(file, delimiter='\t')
		next(reader, None) 	# skip header

		writer = csv.writer(out, delimiter='\t')

		writer.writerow(['File Name', 'Subject Description', 'Motion Description', 'KeyWords', 'Number of Frames', 
    	'Frame Rate', 'Length (s)'])

		for row in reader:
			subj_num = row[0]
			subj_desc = row[1]
			trial_num = row[2]
			motion_desc = row[3]
			framerate = row[4]

			subj_num = '0' + subj_num if len(subj_num) == 1 else subj_num
			trial_num = '0' + trial_num if len(trial_num) == 1 else trial_num

			file_name = subj_num + '_' + trial_num + '.amc'
			file_path = folder_path + subj_num + '/' + file_name

			print(file_name)

			try:
				num_frames = getNumFrames(file_path)
				length = float(num_frames) / float(framerate)

				keys = []
				for k in KEYWORDS:
					if k in motion_desc:
						keys.append(k)

				out_row = [file_name, subj_desc, motion_desc, keys, num_frames, framerate, length]
				writer.writerow(out_row)
			except Exception as e:
				print("ERROR! ") + file_name

def updateMapping():
	tempfile = NamedTemporaryFile(mode = 'w', delete=False)
	with open('cmu_mocap_labels.tsv','r') as file, tempfile as temp:
		reader = csv.reader(file, delimiter='\t')
		writer = csv.writer(temp, delimiter='\t')
		writer.writerow(next(reader, None)) 	# copy header

		for row in reader:
			motion_desc = row[2].lower()
			keys = []
			for k in KEYWORDS:
				if k in motion_desc:
					keys.append(k)
			row[3] = keys
			writer.writerow(row)

	shutil.move(tempfile.name, 'cmu_mocap_labels.tsv')

def readMapping():
	frames = []
	length = []

	with open('cmu_mocap_labels.tsv', 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		next(reader, None) 	# skip header

		for row in reader:
			frames.append(int(row[4]))
			length.append(float(row[6]))

	return (frames, length)

def plotLength():
	frames, length = readMapping()

	plt.figure(1)

	plt.subplot(211)
	plt.title('Number of Frames')
	plt.hist(np.array(frames), bins=100)

	plt.subplot(212)
	plt.title('Length of Sequence (s)')
	plt.hist(np.array(length), bins=100)

	plt.show()

def filterAndCopy():
	with open('cmu_mocap_labels.tsv', 'r') as file, open('filtered_cmu_labels.tsv', 'w') as filtered:
		reader = csv.reader(file, delimiter='\t')
		writer = csv.writer(filtered, delimiter='\t')
		#copy header
		writer.writerow(next(reader, None))

		for row in reader:
			length = float(row[6])
			keywords = row[3]
			if keywords == '[]' or length > 10:
				continue
			keywords = keywords[1:-1].split(",")
			if len(keywords) > 1 and not ((" 'stand'" in keywords or "'stand'" in keywords) and len(keywords) == 2):
				continue
			writer.writerow(row)

def main():
	#parseMapping()
	#plotLength()
	#updateMapping()
	filterAndCopy()

if __name__ == '__main__':
  main()