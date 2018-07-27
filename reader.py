from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
import collections

def reader():
	images = [join('images', f) for f in listdir('images') if isfile(join('images', f))]
	image_list = [np.array(misc.imread(image))[:, 0 : 360, 0 : 3] for image in images] 
	arr = np.array(image_list)
	return arr

def readLabel(start, end):
	file_object  = open("result.txt", "r")
	labels = [int(line.replace('\n', '')) for line in file_object.readlines()]
	arr = np.array(labels)[start:end]
	return arr

