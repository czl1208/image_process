from scipy import misc
import numpy as np
import collections

def rgb2hex(r,g,b):
    hex = "#{:02x}{:02x}{:02x}".format(r,g,b)
    return hex

image = misc.imread('img1.png')
arr = np.array(image)
colors = [];
for row in arr:
	for color in row:
		colors.append(color)

color_list = [rgb2hex(color[0], color[1], color[2]) for color in colors]
counts = collections.Counter(color_list)
new_list = sorted(counts, key=lambda x: -counts[x])
print(new_list[0])