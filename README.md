# Super Resolution Image Reconstruction (Upscaling) with Convolutional Neural Networks

In this small project I tried to build a convolutional neural network to perform image reconstruction (upscaling), increasing the resolution of a 64x64 pixel image to a 128x128 pixel image. The following image shows the input, the model output and the expected input of a test image.

![A frame of a test video](https://github.com/alghul96/CNN_SR_image_reconstruction/blob/master/Results/frame.png)

As one may notice, the results are pretty interesting. The overall loss of the model, infact, was around 0.002.
In the next image we compare the results with the built-in linear interpolation of skimage. It is noticeable that with just few hours of training this simple architecture archived better clarity and details in the face than the previous mentioned method.
More examples like this can be found in the folder results.

![Comparison with Skimage upscaling](https://github.com/alghul96/CNN_SR_image_reconstruction/blob/master/Results/8.png)


## Getting Started

If you would like to reproduce the work I have done, simply follow this paragraph to set everything up on your machine.

### Prerequisites

The project includes numerous external packages. The main are:

- **numpy**: used for reading, indexing, reshaping and saving arrays;
- **keras**: used for building and training the model, requires Theano or TensorFlow;
- **skimage**: for importing the data (not required if you intend to use already provided data);
- **ffmpeg**: used for importing and converting videos in frame (not required if you intend to use already provided data).

Packages can be easily installed via pip or Conda. For example:
```
pip install numpy
```
### Gathering data

Multiple image datasets and videos were used to train the neural network. I personally used the [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) (used also for testing) and the cat and dog recognition dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
For the videos, I took some old samples from Wikimedia Commons. The one provvided comes from [here](https://commons.wikimedia.org/wiki/Category:Videos_of_John_F._Kennedy)

**However**, to reproduce the experiment one can easily start from the ready-made datasets contained in the data folder in the format of compressed arrays.

If you intend to import the data anyway, you can use the script I wrote **data_preparation.py**


### Preparing data

The script contains two simple functions, one for importing images from a folder, one for importing videos.
It automatically scale the pictures to the desired resolutions for the experiment, flatten the pictures from RGB to black and white and return a compressed numpy array.
One should just specify in the function call the path to the new images before running the script.
Similarly, one can import videos. For each frame of the video, in this way, a separate image will be created and stored in a dataset.

Note that the images should be directly placed in the specified folder.
This means that, for datasets such as Face in the Wild, every subdirectory needs to be deleted.
One can easily achieve that any command line with:

```
move D:\rootpath\*\* D:\path
```

## Running the experiment

I am used to jump from a script to the other. Hence, I will try to list all the scripts and the preferred order of execution.
Hopefully, it is clear enough to reproduce everything.

### File Organization

- **data**: folder containing the compressed numpy arrays necessary to the analysis.
- **results**: folder containing some examples of what the neural net is capable of.

The scripts are the following. One should run them in the order they are listed.

- **data_preparation.py**: code for prepare the previous mentioned data. It is not strictly necessary to run it, because the data folder already contains all the necessary materials.
- **model.py**: this is the main script, containing the architecture of the model and the lines necesary for the first training.
- **training.py**: code for training (and continue the training) for the model.
- **testing.py**: code for testing the model and produce those graphs that are contained in the folder **results**
- **testing_video.py**: code for generating from (almost) any video a test video and the predicted result.

The already trained model, lastly, is saved in the file: **model.h5**

### Training and improve training

It is interesting that the network _recognize_ particular features included in the images of the given data.
In other words, it learns how to draw specific patterns included in the training set.
For example, take the following picture of my friend, Joris, and his marvellous cat, Prada:

![He is a nice kitty](https://github.com/alghul96/CNN_SR_image_reconstruction/blob/master/Results/cat.png)

Notice how from the little data on the eye of the animal it tried to reconstruct cat's pupil.
Hence, better results in the face of the animal are achieved if one trains the model on just images of cats, for example.


## Authors

This experiment was fully designed and coded by Gaetano Romano. 


### Contacts:
- **Email**: gaetano.romano.96@gmail.com, gaetano.romano@ucdconnect.ie
- **LinkedIn**: [gaetanoromano96](https://www.linkedin.com/in/gaetanoromano96/)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

