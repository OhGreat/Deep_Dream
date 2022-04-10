
# Deep dream implementation with Tensorflow networks 

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About The Project</a>
      <ul>
        <li><a href="#examples">Examples</a></li>
        <li><a href="deep_dream_as_art">Is Deep Dream art?</a></li>
      </ul>
    </li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#future-work">Future work</a></li>
    <li><a href="#references-and-acknowledgements">References & acknowledgement</a></li> 
  </ol>
 </details>

## About
Deep Dream is a computer vision tool, created by Google's engineer Alexander Mordvintsev, to help us understand how neural networks work. In general, it uses the convolutional layers of Neural Networks to find and enhance patterns in images, thus intentionally creating dream-like psychedelic over-processed images.


### Examples
Examples created with Deep Dream. It is recommended to open the created images in a new tab, to see all the details and patterns created by the algorithm.

**The examples below are generated from a starting input image:**

- These examples were created with lower network layers and smaller octave range in order to create more subtle *dreamifications*, preserving the main shapes and structures.
<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/input/lighthouse.jpg" height="400">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/output/lighthouse.jpeg" height="400">
  <br/>
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/input/selfie.jpg" height="400">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/output/selfie.jpeg" height="400">
</div>
<br/><br/>
<ul>
  <li> These examples on the other hand, are created with deeper layers and more steps per octave. In general, we can observe that the images get a deeper modification, with new structures appearing in the image.
  </li>
</ul>

<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/input/leiden_night.jpg" height="350">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/output/leiden_night.jpeg" height="350">
  <br/>
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/input/christmas_dinner.jpg" height="400">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/output/christmas_dinner.jpeg" height="400">
</div>
<br/><br/>

**The examples below are generated from random noise, allowing the network get <i>'creative'</i> with its representations:**
- We can observe how the noise gets interpreted as leafs and branches of trees, with some shapes recalling animals and birds.

<div align="center">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/input/random_noise.jpeg" height="350">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/output/random_noise.jpeg" height="350">
  <br/>
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/input/random_noise_2.jpeg" height="350">
  <img src="https://github.com/OhGreat/Deep_Dream/blob/main/output/random_noise_2.jpeg" height="350">
</div>
<br/><br/>


<h3 id="deep_dream_as_art"> Is Deep Dream some kind of art?</h3>

*“Google Dream (...) is a tool, not the product, so calling it art would be a little like an artist raising their hand and declaring their paintbrush art because they were so happy with the way they used it lay paint on a canvas.” <p align="right">- Paddy Johnson</p>* 


Although Deep Dream cannot exactly be considered art, its applications are intriguing and representative of our brain as explained by Memo Akten in the citation below.

<br/>

*"It might look like Deep Dream is generating say, sparrow’s faces in clouds, but what it is actually doing is generating patterned noise, which our brains try to find meaning in. It creates just enough of a sparrow’s head in a cloud, so that our brains find the rest. Visually, our minds and Deep Dream are doing exactly the same thing. It’s such a perfect mirror. I love that conceptual aspect." <p align="right">- Memo Akten</p>*


### About the implementation
This work is mainly based on the tiled implementation of the Deep Dream Tensorflow tutorial found <a href="https://www.tensorflow.org/tutorials/generative/deepdream">here</a>. Only subtle details of the implementation have been changed. A wrapper has been constructed to choose from various models and run experiments on the algorithm in various configurations.<br/>



## Requirements
<ul>
  <li>Python 3</li>
  <li>numpy >= 1.19</li>
  <li>PIL</li>
  <li>Matplotlib</li>
  <li>Tensorflow >= 2.5.0</li>
</ul>
CUDA is recommended

## Usage

The main file to run experiments is the ***deep_dream.py*** file in the main directiry. An accurate description of all the arguments is available below.

The ***deep_dream.sh*** script in the main directory can be used to set parameters and run experiments directly from a bash terminal.

### Tunable parameters:
- ***-img***: Path of the input image to use for the experiment. It can be set to *'random_noise'* (set as default) in order to create a random normal distributed noise to pass to the deep dream model.
- ***-img_out***: Name for the output image. Default is *'output'*.
- ***-img_max_dim***: Resize input image to a maximum dimension, which will be used to create the output image. Aspect ration will be kept. Default value is *1024*.
- ***-ts***: Size of the tile for the image roll. Should be a number between 0 and *img_max_dim*. Default is *512*.
- ***-ss***: Step size to update our image at every iteration. Some good values for this parameter are between 0.001 and 0.1 . Default value is 0.01 .
- ***-osc***: Octave scale. Scale parameter that determines the size of the image at every octave.
- ***-or***: Octaves range. This parameter takes as input two values, representing the starting and ending indexes of the octave range. Each value of the octave is multiplied by *osc* to get the size of the image for the current octave. Choosing negative values will make the image smaller than the original, resulting in filters that have more impact to the global structure and shapes. Higher octave values on the other hand, will make the image bigger than the original, preserving the overall structure and shapes of the image while making the amplifications appear more like adding details. The default values are [-4, 2] creating the octave range [-4, -3, -2, -1, 1] .
- - ***-os***: Number of iterations to run for each octave.
- ***-m***: String parameter representing the base model to use for the deep dream. Possible values for this parameters is represented by the models currently implemented. For now this parameter can be set to either *'inceptionV3'* or *'inceptionResNet'* .
- ***-ml***: Range of the model layers to use for the deep dream. If left blank or wrongly chosen, the application will ask you to complete this parameter during runtime, after informing the user of the available range of layers. For *'inceptionResNet'* model the available range is from 0 to 43, while for the *'inceptionV3'* it is from 0 to 15 .


## Future Work
- ~~Add more args for more detailed control<br/>~~
- ~~Create bash script to run experiemnts.~~

## References and acknowledgements:

- https://www.tensorflow.org/tutorials/generative/deepdream

- https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

- https://hackernoon.com/deep-dream-with-tensorflow-a-practical-guide-to-build-your-first-deep-dream-experience-f91df601f479

- https://www3.cs.stonybrook.edu/~cse352/T12talk.pdf

- https://www.youtube.com/watch?v=BsSmBPmPeYQ&ab_channel=Computerphile

- https://www.ted.com/talks/blaise_aguera_y_arcas_how_computers_are_learning_to_be_creative?language=en#t-898371

- https://www.theguardian.com/artanddesign/2016/mar/28/google-deep-dream-art
