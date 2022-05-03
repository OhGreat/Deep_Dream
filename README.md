
# Deep dream implementation using Tensorflow

 <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/deep_dream.gif" width="100%" > 

**Table of Contents**
<ol align="left">
  <li><a href="#about">About</a></li>
  <ul>
   <li><a href="#examples">Examples & Observations</a></li>
   <li><a href="#deep_dream_as_art">Is Deep Dream considered art?</a></li>
  </ul>
  <li><a href="#prerequisites">Prerequisites</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#future-work">Future work</a></li>
  <li><a href="#references-and-acknowledgements">References & acknowledgement</a></li> 
</ol>
<br/>

## About
Deep Dream is a computer vision tool, created by Google's engineer Alexander Mordvintsev, to help us understand how neural networks work. It uses the convolutional layers of Neural Networks to find and enhance patterns in images, thus intentionally creating dream-like, psychedelic, over-processed images.This work is mainly based on the tiled implementation of the Deep Dream Tensorflow tutorial found <a href="https://www.tensorflow.org/tutorials/generative/deepdream">here</a>. Subtle details of the implementation have been changed and it is currently under work. In addition, a wrapper and a script have been constructed to choose from various models and run experiments on the algorithm with different configurations.<br/>


<h2 id="examples"> Examples & Considerations</h3>
In this section you will find a collection of images created with this Deep Dream framework, together with a few observations. It is recommended to open the created images in a new tab, to see all the details and patterns created by the algorithm.
<br/><br/>

**The examples below have been generated from a starting input image:**

- These examples were created with lower network layers and smaller octave range in order to create more subtle *dreamifications*, preserving the shapes and structures of the original image.

 
<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/input/lighthouse_reshaped.jpeg" height="400">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/output/lighthouse.jpeg" height="400">
  <br/>
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/input/selfie_reshaped.jpeg" height="400">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/output/selfie.jpeg" height="400">
</div>
<br/><br/>
<ul>
  <li> The following examples on the other hand, were created with deeper layers and more steps per octave. In general, we can observe that the images got a deeper modification, with new shapes and structures appearing in the image.
  </li>
</ul>

<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/input/granada_reshaped.jpeg" width="380">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/output/granada.jpeg" width="380">
  
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/input/snow_reshaped.jpeg" width="380">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/output/snow.jpeg" width="380">
  <br/>
</div>
<br/><br/>

**The examples below have been generated from random noise, allowing the network get <i>'creative'</i> with its representations:**
- We can observe how the noise gets interpreted as leafs and branches of trees, with some shapes recalling animals and birds.

<br/>
<div align="center">
  Example input noise:
</div>
<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/input/random_noise.jpeg" width="200"> 
</div>
<br/><br/>
<div align="center">
  Example outputs:
</div>
<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/output/random_noise_1.jpeg" width="350">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/readme_files/output/random_noise_2.jpeg" width="350">
</div>
<br/>

<h3 id="deep_dream_as_art"> Is Deep Dream considered art?</h3>

*“Google Dream (...) is a tool, not the product, so calling it art would be a little like an artist raising their hand and declaring their paintbrush art because they were so happy with the way they used it lay paint on a canvas.” <p align="right">- Paddy Johnson</p>* 


Although Deep Dream cannot exactly be considered art, its applications are intriguing and representative of our brain as explained by Memo Akten from the citation below.
<br/>

*"It might look like Deep Dream is generating say, sparrow’s faces in clouds, but what it is actually doing is generating patterned noise, which our brains try to find meaning in. It creates just enough of a sparrow’s head in a cloud, so that our brains find the rest. Visually, our minds and Deep Dream are doing exactly the same thing. It’s such a perfect mirror. I love that conceptual aspect." <p align="right">- Memo Akten</p>*
<br/>

## Prerequisites

### Required

`Python3` version 3.8 ~ 3.10 is required, with the following packages installed:
- `numpy`
- `Pillow`
- `Matplotlib`
- `Tensorflow` >= 2.8.0


### Recommended
- `CUDA` (to run Tensorflow on GPU)


## Usage

The main file to run experiments is the `deep_dream.py` file in the main directiry. An accurate description of all the arguments is available below.

The script `deep_dream.sh` in the main directory is also available as an example to set parameters and run experiments.

### Tunable parameters:
- `-img`: Path of the input image to use for the experiment. It can be set to *'random_noise'* (set as default) in order to create a random normal distributed noise to pass to the deep dream model.
- `-img_out`: Name for the output image. Default is *'output'*.
- `-img_max_dim`: Resize input image to a maximum dimension, which will be used to create the output image. Aspect ration will be kept. Default value is *1024*.
- `-save_input`: When used, the flag will save the input image resized to the maximum dimension defined with the previous flag *-img_max_dim*.
- `-ts`: Size of the tile for the image roll. Should be a number between 0 and *img_max_dim*. Default is *512*.
- `-ss`: Step size to update our image at every iteration. Some good values for this parameter are between 0.001 and 0.1 . Default value is *0.01* .
- `-osc`: Octave scale. Scale parameter that determines the size of the image at every octave. Its default value is *1.2* .
- `-or`: Octaves range. This parameter takes as input two values, representing the starting and ending indexes of the octave range. Each value of the octave is multiplied by *osc* to get the size of the image for the current octave. Choosing negative values will make the image smaller than the original, resulting in filters that have more impact to the global structure and shapes. Higher octave values on the other hand, will make the image bigger than the original, preserving the overall structure and shapes of the image while making the amplifications appear more like adding details. The default values are *[-4, 2]* creating the octave range [-4, -3, -2, -1, 0, 1] .
- `-os`: Number of iterations to run for each octave. The default value is *80*.
- `-m`: String parameter representing the base model to use for the deep dream. Possible values for this parameters is represented by the models currently implemented. For now this parameter can be set to either *'inceptionV3'* or *'inceptionResNet'* .
- `-ml`: Range of the model layers to use for the deep dream. If left blank or wrongly chosen, the application will ask you to complete this parameter during runtime, after informing the user of the available range of layers. For *'inceptionResNet'* model the available range is from 0 to 43, while for the *'inceptionV3'* it is from 0 to 15 .
<br/>


## Future Work

- ~~Add more args for more detailed control<br/>~~
- ~~Create bash script to run experiemnts.~~
- Anneal value of tile size and other params between steps.
<br/>

## References and acknowledgements:

The following links were used as reference to build the framework:

- https://www.tensorflow.org/tutorials/generative/deepdream

- https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

- https://hackernoon.com/deep-dream-with-tensorflow-a-practical-guide-to-build-your-first-deep-dream-experience-f91df601f479

- https://www3.cs.stonybrook.edu/~cse352/T12talk.pdf

- https://www.youtube.com/watch?v=BsSmBPmPeYQ&ab_channel=Computerphile

- https://www.ted.com/talks/blaise_aguera_y_arcas_how_computers_are_learning_to_be_creative?language=en#t-898371

- https://www.theguardian.com/artanddesign/2016/mar/28/google-deep-dream-art
