import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("""
İbrahim Halil Bayat 
Department of Electronics and Communication Engineering 
İstanbul Technical University 
İstanbul, Turkey

Facial Expression Recognition 
0 - Angry 
1 - Disgust 
2 - Fear 
3 - Happy 
4 - Sad
5 - Surprise 
6 - Neutral 
""")

my_num_classes = 7
my_batch_size = 256
my_epochs = 5

with open("fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("Number of Instances. ", num_of_instances)
print("Instance Length: ", len(lines[1].split(",")[1].split(" ")))

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1, num_of_instances):

    emotion, img, usage = lines[i].split(",")
    val = img.split(" ")
    pixels = np.array(val, 'float32')
    emotion = keras.utils.to_categorical(y=emotion, num_classes=my_num_classes)

    if 'Training' in usage:
        y_train.append(emotion)
        x_train.append(pixels)
    elif 'PublicTest' in usage:
        y_test.append(emotion)
        x_test.append(pixels)

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

print("\n ------------- Normalizing X_Train and X_Test by 255 -----------------")

x_train /= 255
x_test /= 255

print("\n ---------------- Reshaping X_Train and X_Test sets  by 48, 48, 1 ---------------")

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print("Train Samples: ", x_train.shape[0])
print("Test Samples: ", x_test.shape[0])

print("\n ------------------- Creating the Model -------------------")
model = Sequential()
print("Model is selected as sequential.")

model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
print("1st Layer: 64 Convolution filters and Max Pooling.")

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
print("2nd Layer: 64 Convolution filters x 2 and Average Pooling")

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
print("3rd Layer: 128 Convolution filters x 2 and Average Pooling")

model.add(Flatten())
print("Model is flattened")

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(my_num_classes, activation='softmax'))

print("\n ----------- Batch ---------------------")
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=my_batch_size)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

fit = True

if fit == True:
    #model.fit_generator(x_train, y_train, epochs=my_epochs)  # Training for the whole class
    model.fit_generator(train_generator, steps_per_epoch=my_batch_size, epochs=my_epochs)
else:
    model.load_weights('weights.h5')

original_image = image.load_img("pp.png")
img = image.load_img("pp.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x /= 255

custom = model.predict(x)

objects = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
y_pos = np.arange(len(objects))

plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel("Percent")
plt.title("Emotion")
plt.show()

x = np.array(x, 'float32')
x = x.reshape([48, 48])
plt.axis('off')
plt.gray()
plt.imshow(original_image)
plt.show()