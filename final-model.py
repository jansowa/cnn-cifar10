import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.layers import BatchNormalization
import numpy as np
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from os import listdir
from matplotlib import image

num_classes = 10

# big batch_size can cause generalization problems:
# https://stats.stackexchange.com/a/236393

# tools for resources monitoring:
# nvidia-smi -l 1
# free -m -l -s 1

# batch_size = 128 could be better, but I haven't got enough gpu memory
batch_size = 64
epochs = 1000
img_rows, img_cols = 32, 32

# load all images in a directory

X = list()
filesnames = listdir('train')


def sortKey(filename):
    dotindex = filename.index(".")
    return int(filename[:dotindex])


filesnames.sort(key=sortKey)
for filename in filesnames:
    # load image
    img_data = image.imread('train/' + filename)
    # store loaded image
    X.append(img_data)
print("Number of loaded images:", len(X))
X = np.array(X)

y_raw = pd.read_csv("trainLabels.csv")["label"]
le = preprocessing.LabelEncoder()
le.fit(y_raw)
y_le = le.transform(y_raw)
y = keras.utils.np_utils.to_categorical(y_le, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X.shape)
print(X_train.shape)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

model = Sequential()

# https://datascience.stackexchange.com/questions/102483/difference-between-relu-elu-and-leaky-relu-their-pros-and-cons-majorly
# relu -> x if x>0 else 0
# leaky relu -> x if x>0 else 0.01x
# elu -> x if x>0 else 0.01 * (exp(x) - 1)
activation_function = "leaky_relu"
if activation_function == "leaky_relu":
    activation_function = keras.layers.LeakyReLU(alpha=0.01)

# https://stackoverflow.com/questions/36243536/what-is-the-number-of-filter-in-cnn
# "filters" param sets the maximum number of filters in layer
# each filter creates new feature map (much higher complexity!)
conv_two_exp = 7
first_layer_filters = 2 ** conv_two_exp
kernel_unit = 3
kernel_size = (kernel_unit, kernel_unit)
kernel_initializer = 'he_uniform'

model.add(Conv2D(first_layer_filters, kernel_size=kernel_size,
                 activation=activation_function,
                 input_shape=input_shape, kernel_initializer=kernel_initializer))
# https://www.baeldung.com/cs/batch-normalization-cnn
# BatchNormalization - normalizes data between mini-batches. It allows using higher learning rate.
# Subtract mean of neurons output and divide by standard deviation.
# "each feature map will have a single mean and standard deviation, used on all the features it contains"
model.add(BatchNormalization())
model.add(
    Conv2D(first_layer_filters, kernel_size, activation=activation_function, kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
dropout1 = 0.07500704262884543
model.add(Dropout(dropout1))

# 2x more filters in each next Conv2D layer
model.add(
    Conv2D(first_layer_filters * 2, kernel_size, activation=activation_function, kernel_initializer=kernel_initializer))

# MaxPooling2D Reduces number of trainable parameters
model.add(MaxPooling2D(pool_size=(2, 2)))
dropout2 = 0.1349281039679955
model.add(Dropout(dropout2))
model.add(
    Conv2D(first_layer_filters * 4, kernel_size, activation=activation_function, kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(
    Conv2D(first_layer_filters * 8, kernel_size, activation=activation_function, kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
dropout3 = 0.12798787648693688
model.add(Dropout(dropout3))
model.add(Flatten())
dense_two_exp = 9
dense_units = 2 ** dense_two_exp
model.add(Dense(dense_units, activation=activation_function))
model.add(BatchNormalization())
dropout4 = 0.23270556518901153
model.add(Dropout(dropout4))
model.add(Dense(dense_units / 4, activation=activation_function))
model.add(BatchNormalization())
dropout5 = 0.4304300824505963
model.add(Dropout(dropout5))
model.add(Dense(num_classes, activation='softmax'))

# Adam optimizer tunes learning rate in next epochs
learning_rate = 0.0009766910468362844
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2851945624723854,  # Randomly zoom image
    shear_range=0.42664072982759615,  # shear angle in counter-clockwise direction in degrees
    width_shift_range=0.14234862104307688,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.12363487226276715,  # randomly shift images vertically (fraction of total height)
    vertical_flip=False,
    horizontal_flip=True)  # randomly flip images

datagen.fit(X_train)

earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                        mode="max", patience=100,
                                        restore_best_weights=True)
print(model.summary())

hist = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                 epochs=epochs,
                 verbose=1,
                 validation_data=(X_test, y_test),
                 callbacks=[earlystopping])

model.save('cnn-model')
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
