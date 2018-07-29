from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
import collections

def reader(path):
	images = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
	image_list = [np.array(misc.imread(image))[:, 0 : 360, 0 : 3] for image in images] 
	image_list = [image if image.shape[1] >= 360 
					else np.concatenate(
						(image, 
						 np.zeros((image.shape[0], 
						 			360 - image.shape[1], 
						 			image.shape[2]))), 1) 
					for image in image_list]
	arr = np.array(image_list)
	return arr

def readLabel(start, end):
	file_object  = open("result.txt", "r")
	labels = [int(line.replace('\n', '')) for line in file_object.readlines()]
	arr = np.array(labels)[start:end]
	return arr