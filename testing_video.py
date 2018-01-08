import pylab
import imageio
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, concatenate_images
from keras.models import load_model
import matplotlib.pyplot as plt

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

print("Reading Video")
x, y = read_video("data/kennedy.webm", (64, 64), (128, 128))
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

print("Loading model")
model = load_model("model.h5")
predictions = model.predict(x)

input_vid = imageio.get_writer("Results/input_video.mp4", "ffmpeg")
output_vid = imageio.get_writer("Results/output_video.mp4", "ffmpeg")

for i in range(0, len(x) - 1):
    input_vid.append_data(x[i, :, :, 0])
    output_vid.append_data(predictions[i, :, :, 0])

input_vid.close()
output_vid.close()

# plotting some frames for example
# frame example
i = 600
plt.subplot(221)
plt.imshow(x[i, :, :, 0], cmap="Greys_r")
plt.title("Low res. input image")
plt.axis("off")

# target image
plt.subplot(222)
plt.imshow(y[i, :, :, 0], cmap="Greys_r")
plt.title("High res. target image")
plt.axis("off")

# rescaled image
plt.subplot(223)
resized = resize(x[i, :, :, 0], (128, 128), mode="constant")
plt.imshow(resized, cmap="Greys_r")
plt.title("Rescaled high res. image")
plt.axis("off")

# model2 predicted image
plt.subplot(224)
plt.imshow(predictions[i, :, :, 0], cmap="Greys_r")
plt.title("Model high res. reconstruction")
plt.axis("off")

plt.show()