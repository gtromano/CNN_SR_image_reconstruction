from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Flatten, Reshape, Dropout, Conv2DTranspose
from keras.callbacks import ModelCheckpoint
import numpy as np

# loading previously created data
print("Loading previously created data...")
data = np.load("data/faceimages.npz")

input_dim = data["x_test"].shape[1:]
output_dim = data["y_test"].shape[1:]


# Building the model
model = Sequential()

# convolution layers
model.add(Conv2D(1, (1, 1), data_format="channels_last", input_shape=input_dim))
model.add(Conv2D(2, (3, 3)))
model.add(Conv2D(3, (4, 4)))
model.add(Dropout(.05))

# Transpose convolution layers (Deconvolution)
model.add(Conv2DTranspose(3, (3, 3)))
model.add(Conv2DTranspose(2, (5, 5)))
model.add(Conv2DTranspose(1, (8, 8)))
model.add(Dropout(.1))

# Fully connected layers
model.add(Flatten())
model.add(Dense(np.prod(output_dim)))
model.add(Reshape(output_dim)) # scaling to the output dimension
model.add(Activation("linear")) # using a "soft" activation

model.compile(optimizer = "adam", loss = "mse")
print(model.summary())


# fitting the model
print("Fitting the model...")
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(data["x_train"], data["y_train"],
          batch_size=200,
          epochs=40,
          validation_data=(data["x_test"], data["y_test"]),
          callbacks = callbacks_list)