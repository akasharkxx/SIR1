from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.utils import np_utils
import h5py

# dog_img_path = '/home/r1phunt3r/Downloads/puppy-dog.jpg'
# dog_img = Image.open(dog_img_path)
# dog_img.show()

# made the above more dynamic
# disp_img_path = input("Enter Image path : ")
# disp_img = Image.open(disp_img_path)
# disp_img.show()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# index = int(input("Enter an image index : "))
# disp_img = X_train[index]
# disp_label = y_train[index][0]
#
# from matplotlib import pyplot as plt
# red_img = Image.fromarray(disp_img)
# red, green, blue = red_img.split()
#
# plt.imshow(blue, cmap="Blues")
# plt.show()
# print(labels[disp_label])

new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_test /= 255
new_X_train /= 255
new_y_train = np_utils.to_categorical(y_train)
new_y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
# recommended epochs = 10
model.fit(new_X_train, new_y_train, epochs=100, batch_size=32)

model.save('trained_model.h5')