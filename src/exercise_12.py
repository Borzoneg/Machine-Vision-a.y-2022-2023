import numpy as np
import matplotlib.pyplot as plt
import h5py

train_dataset = h5py.File('../inputFiles/NeuralNetwork/train_catvnoncat.h5', "r")
test_dataset = h5py.File('../inputFiles/NeuralNetwork/test_catvnoncat.h5', "r")

train_x = np.array(train_dataset["train_set_x"][:])
train_y = np.array(train_dataset["train_set_y"][:])
test_x = np.array(test_dataset["test_set_x"][:])
test_y = np.array(test_dataset["test_set_y"][:])

classes = np.array(test_dataset["list_classes"][:])
print(classes)

train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

print ("Train X shape[nr_images, width, height, nr_image]: " + str(train_x.shape)) 
print ("Train Y shape[dim_label, nr_image]: " + str(train_y.shape))
print ("Test X shape [nr_images, width, height, nr_image]: " + str(test_x.shape))
print ("Test Y shape [dim_label, nr_image]: " + str(test_y.shape))

plt.imshow(train_x[0])