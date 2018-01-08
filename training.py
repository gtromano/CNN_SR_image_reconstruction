from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np

# importing data
data = np.load("data/faceimages.npz")

# loading model
model = load_model("model.h5")
print(model.summary())

# start training
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(data["x_train"], data["y_train"],
          batch_size=200,
          epochs=30,
          validation_data=(data["x_test"], data["y_test"]),
          #validation_split=0.001,
          callbacks = callbacks_list)
