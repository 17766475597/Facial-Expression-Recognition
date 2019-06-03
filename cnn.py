from __future__ import print_function
import numpy as np

filename = 'fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#得到数据集中的值 X为对应的图片向量 Y为图片对应的表情标签
def getData(filename):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    flag = True
    for line in open(filename):
        if flag:
            flag = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y


X, Y = getData(filename)
num_class = len(label_map)

# 输出7类表情各自的个数
def balance_class(Y):
    num_class = set(Y)
    count_class = {}
    for i in range(len(num_class)):
        count_class[i] = sum([1 for y in Y if y==i])

    return count_class

balance = balance_class(Y)

print(balance)
N, D = X.shape
print(N)
X = X.reshape(N, 48, 48, 1)

#训练集与测试集分离
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
print(np.arange(num_class))
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
print(y_train)
print(y_test)

from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

batch_size = 128
epochs = 100

#CNN模型搭建
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1st Convolution
    model.add(Conv2D(64,(3,3), border_mode='same', input_shape=(48, 48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd Convolution
    model.add(Conv2D(128,(5,5), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd Convolution
    model.add(Conv2D(256, (3, 3), border_mode='same'))
    model.add(Conv2D(256, (3, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # 4th Convolution
    model.add(Conv2D(512, (3, 3), border_mode='same'))
    model.add(Conv2D(512, (3, 3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    # Fully connected layer 1st layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully connected layer 2nd layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_class, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


def baseline_model_saved():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


is_model_saved = True

if (is_model_saved == False):
    model = baseline_model()

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")
else:
    print("Load model from disk")
    model = baseline_model_saved()

score = model.predict(X_test)
print(score)
print(model.summary())

new_X = [np.argmax(item) for item in score]
y_test2 = [np.argmax(item) for item in y_test]

accuracy = [(x == y) for x, y in zip(new_X, y_test2)]
print(" Accuracy on Test set : ", np.mean(accuracy))


from PIL import Image
import numpy as np

def myResize(filename, width, height):
    image = Image.open(filename).convert('L')
    image_resize = image.resize((width, height), Image.ANTIALIAS)
    image_resize.save("zhouzh_gray.jpg")
    image_resize.close()


filename = "zhouzh.jpg"
width = 48
height = 48
myResize(filename,width,height)
image = Image.open('zhouzh_gray.jpg')
image_arr = np.array(image)

print(image_arr)
image_arr = image_arr.reshape(1,48,48,1)
score_image = model.predict(image_arr)
print(score_image)
a = np.array(score_image[0])
idx = 0
for i in range(7):
    if a[i]>a[idx]:
        idx = i

print(label_map[idx])
import matplotlib.pyplot as plt
image1 = Image.open('zhouzh_gray.jpg')
plt.imshow(image1)
plt.show()

image2 = Image.open('zhouzh.jpg')
plt.imshow(image2)
plt.title(label_map[idx],fontsize='x-large')
plt.show()