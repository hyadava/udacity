import pickle
import numpy as np
import math
import cv2
import csv
import os, sys
from scipy import ndimage

from sklearn.cross_validation import train_test_split


img_folder = "./trdata/IMG"  
label_files = {'./trdata_straight/driving_log.csv':  0, './trdata_zigzag/driving_log.csv':  1, './trdata_positive/driving_log.csv': 2}

print('loading images from ', img_folder, ', labels from ', label_files)


def crop_img(img):
	#img = img[60:, :, :]
	img = img[70: -20, 20:-20, :]
	#img = img[60: -20, :, :]
	return img


#find image size
img_height =70
img_width =280 
img_channels = 3

img_list = list()
label_dataset = list()

def load_data(label_file, record_type):
	loaded = 0
	skipped = 0
	with open(label_file) as f:
		reader = csv.reader(f)
		for i in reader:
			if(record_type == 0):
				img_list.append(i[0])
				label_dataset.append(i[3])
				loaded += 1
			elif (record_type == 1 or record_type == 2):
				if(float(i[3]) != 0):
					#for c in range(2):
						img_list.append(i[0])
						label_dataset.append(i[3])
						loaded += 1
				else:
					skipped += 1
	print('loaded ', loaded, ' skipped ', skipped)


skipped = 0
for k, v in label_files.items():
	print('loading ', k)
	load_data(k, v)

label_dataset = np.array(label_dataset, dtype=np.float32)

#image_files = os.listdir(img_folder)
img_dataset = np.ndarray(shape=(len(img_list), img_height, img_width, img_channels),
                         dtype=np.float32)
pixel_depth = 255
num_images = 0
for im in img_list:
	image_file = im
	if(os.path.isdir(image_file) == True):
		continue

	image = ndimage.imread(image_file).astype(float)

	cropped = crop_img(image)

	img_dataset[num_images, :, :] = cropped
	num_images = num_images + 1


print('image_dataset ', img_dataset.shape)
print('label_dataset ', label_dataset.shape)

#generate randomized train and test sets
X_train, X_test, y_train, y_test = train_test_split(img_dataset, label_dataset, test_size=0.20, random_state=71)
X_train = X_train.reshape(X_train.shape[0], img_height, img_width, 3)
X_test = X_test.reshape(X_test.shape[0], img_height, img_width, 3)

print('X_train ', X_train.shape)
print('X_test ', X_test.shape)

#save data in a pickle file

import tables

comp_filter = tables.Filters(complib='zlib', complevel=5)

h5file = tables.open_file('./trdata/drive_data.h5', mode='w', title="drive_data", filters=comp_filter)
root = h5file.root
h5file.create_carray(root, "train_data", obj=X_train)
h5file.create_carray(root, "train_labels", obj=y_train)
h5file.create_carray(root, "test_data", obj=X_test)
h5file.create_carray(root, "test_labels", obj=y_test)
h5file.close()
