from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

# loading data
print("Loading previously created data...")
data = np.load("data/faceimages.npz")

# loading the previous model
print("Loading Model")
model = load_model("model.h5")

# making a prediction on test dataset
y_pred = model.predict(data["x_test"])


# plotting some random images of the dataset
for i in np.random.random_integers(0, 100, size=10):

    # input image
    plt.subplot(221)
    plt.imshow(data["x_test"][i,:,:,0], cmap="Greys_r")
    plt.title("Low res. input image")
    plt.axis("off")

    # target image
    plt.subplot(222)
    plt.imshow(data["y_test"][i,:,:,0], cmap="Greys_r")
    plt.title("High res. target image")
    plt.axis("off")

    # rescaled image
    plt.subplot(223)
    resized = resize(data["x_test"][i,:,:,0], (128, 128), mode="constant")
    plt.imshow(resized, cmap="Greys_r")
    plt.title("Rescaled high res. image")
    plt.axis("off")

    # model2 predicted image
    plt.subplot(224)
    plt.imshow(y_pred[i,:,:,0], cmap="Greys_r")
    plt.title("Model high res. reconstruction")
    plt.axis("off")

    plt.show()