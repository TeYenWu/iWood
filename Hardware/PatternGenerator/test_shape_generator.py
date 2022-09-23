import numpy as np
import math
import glob
import cv2

minimum_width = 50
minimum_height = 50
# matrix_size = 150


def generate_rect(filename, width, height):
	matrix = np.ones((height, width))
	np.savetxt(filename, matrix, delimiter = ',', fmt='%1u')

def generate_circle(filename, width, height, diameter):
	matrix = np.zeros((height, width))
	for i in range(height):
		for j in range(width):
			if (i - height/2) * (i - height/2) + (j - width/2) * (j - width/2) < diameter/2 * diameter/2:
				matrix[i][j] = 1
	np.savetxt(filename, matrix, delimiter = ',', fmt='%1u')

def generate_from_image(image_file, filename, width, height):
	image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
	print(image.shape)
	matrix = np.zeros((height, width))
	height_scale = height/image.shape[0]
	width_scale = width/image.shape[1]
	for i in range(height):
		for j in range(width):
			image_i = int(i/height_scale)
			image_j = int(j/width_scale)
			if image_i < image.shape[0] and image_j < image.shape[1] and image[image_i][image_j][3] != 0:
				matrix[i][j] = 1

	np.savetxt(filename, matrix, delimiter = ',', fmt='%1u')

if __name__ == '__main__':

	# generate_rect('./test_shapes/rect.csv', minimum_width, minimum_height)
	# generate_circle('./test_shapes/circle.csv', minimum_width, minimum_height, minimum_width)
	for filename in glob.glob('./test_shape_images/*.png'):
	    # print(filename)
	    # break 
	    image_name = filename.split('/')[-1].split('.')[0]
	    print(image_name)
	    generate_from_image(filename, './test_shapes/' + image_name + '.csv', minimum_width, minimum_height)