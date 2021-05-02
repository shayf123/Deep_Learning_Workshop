import os
import numpy as np
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Input

image_dim = (256, 256)
data_path = "C:\\Users\\nbloc\\Desktop\\Data"

# train_dirs, test_dirs: list of indexes from which videos to take data.
# train_batch, test_batch: take frames with jumps of the batches sizes.
# TODO: COMPLETE FOR TEST
def load_data(train_dirs, test_dirs, train_batch, test_batch):
    train_images = []
    for index in train_dirs:

        dir_path = data_path + "\\" + str(index) + "\\frames" + str(index)
        num_of_frames = len(os.listdir(dir_path))

        for frame_counter in range(0, num_of_frames, train_batch):
            image_path = dir_path + "\\frame" + str(frame_counter) + ".jpg"
            train_images.append(cv2.resize(cv2.imread(image_path), image_dim))

    train_images = np.array(train_images)
    train_images = train_images / 255
    print(train_images[0].shape)
    train_images = train_images.reshape((len(train_images), np.prod(train_images[0].shape)))

    return train_images

def vector_to_image(path, vector):
    cv2.imwrite(path, vector.reshape((256, 256, 3)))

def model(input_size, hidden_size):
    x = Input(shape=input_size)
    h = Dense(hidden_size, activation='relu')(x)
    r = Dense(input_size[0], activation='relu')(h)

    autoencoder = Model(inputs=x, outputs=r)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


train_images = load_data([1,2,3,4,5], [], 50, 3)
input_size = train_images[0].shape
print(input_size)
#hidden_size = 585
hidden_size = 1170

autoencoder = model(input_size, hidden_size)

epochs = 5
batch_size = 30

history = autoencoder.fit(train_images, train_images, batch_size=batch_size, epochs=epochs, verbose=1)

test_image = cv2.resize(cv2.imread("C:\\Users\\nbloc\\Desktop\\frame285.jpg"), (256, 256))

test_images = [test_image]
test_images = np.array(test_images)
test_images = test_images / 255
test_images = test_images.reshape((len(test_images), np.prod(test_images[0].shape)))

predicted_images = autoencoder.predict(test_images)
predicted_images = predicted_images * 255
predicted_images = predicted_images.astype(int)
vector_to_image("C:\\Users\\nbloc\\Desktop\\predictedframe285.jpg", predicted_images[0])

# img = vector_to_image(train_images[0]) * 255
# img = img.astype(int)
# cv2.imwrite("C:\\Users\\nbloc\\Desktop\\output285.jpg", img)





