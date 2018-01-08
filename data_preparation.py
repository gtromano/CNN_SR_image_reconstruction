from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, concatenate_images
import os
import numpy as np

# function for importing images from a folder
def load_images(path, input_size, output_size):
    x_ = []
    y_ = []
    counter, totalnumber = 1, len(os.listdir(path))
    for imgpath in os.listdir(path):
        if counter % 100 == 0:
            print("Importing image %s of %s (%s%%)" %(counter, totalnumber, round(counter/totalnumber*100)))
        y = imread(path + "/" + imgpath)
        y = rgb2gray(resize(y, output_size, mode="constant"))
        x = resize(y, input_size, mode="constant")
        x_.append(x)
        y_.append(y)
        counter += 1
    return concatenate_images(x_), concatenate_images(y_)

# function for importing a video, frame by frame
def read_video(filepath, input_size, output_size):
    vid = imageio.get_reader(filepath,  "ffmpeg")
    video_len = vid.get_length()
    counter, totalnumber = 1, video_len
    y_ = []
    x_ = []
    for i in range(0, video_len - 1):
        if counter % 100 == 0:
            print("Importing frame %s of %s (%s%%)" % (counter, totalnumber, round(counter / totalnumber * 100)))
        y_frame = resize(vid.get_data(i), output_size, mode="constant")
        y_frame = rgb2gray(y_frame)
        x_frame = resize(y_frame, input_size, mode="constant")
        y_.append(y_frame)
        x_.append(x_frame)
        counter += 1
    return concatenate_images(x_), concatenate_images(y_)

# defining input and output size
input_size = (64, 64)
output_size = (128, 128)

# loading and reshaping train set
x_train, y_train = load_images("D:\\Users\Pc\Pictures\python\imagedata\\train_faces", input_size, output_size)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
print(x_train.shape, y_train.shape)

# loading and reshaping validation set
x_test, y_test = load_images("D:\\Users\Pc\Pictures\python\imagedata\\test_faces", input_size, output_size)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
print(x_test.shape, y_test.shape)

# saving the data in arrays
print("Creating a compressed dataset...")
np.savez_compressed("image",
                    x_train = x_train,
                    y_train = y_train,
                    x_test = x_test,
                    y_test = y_test)
