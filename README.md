
# Deep dream implementation with Tensorflow networks 
***Work in progress.*** <br/>

## About
This work is mainly based on the tiled implementation of the Deep Dream Tensorflow tutorial found <a href="https://www.tensorflow.org/tutorials/generative/deepdream">here</a>. <br/>



## Requirements
<ul>
  <li>Python 3</li>
  <li>numpy >=1.19</li>
  <li>Tensorflow</li>
  <li>CUDA is recommended</li>
</ul>

## Usage

The main file to run experiments is the ***deep_dream.py*** file in the main directiry. An accurate description of all the arguments is available below.

The ***deep_dream.sh*** script in the main directory can be used to set parameters and run experiments directly from a bash terminal.

### Tunable parameters:
- ***-img***: Path of the input image to use for the experiment. It can be set to *'random_noise'* (set as default) in order to create a random normal distributed noise to pass to the deep dream model.
- ***-img_out***: Optional parameter representing the name for the output image. Default is *'output'*.


## Future Work
- ~~Add more args for more detailed control<br/>~~
- ~~Create bash script to run experiemnts.~~

## References:

- https://www.tensorflow.org/tutorials/generative/deepdream

- https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

- https://hackernoon.com/deep-dream-with-tensorflow-a-practical-guide-to-build-your-first-deep-dream-experience-f91df601f479

- https://www3.cs.stonybrook.edu/~cse352/T12talk.pdf

- https://www.youtube.com/watch?v=BsSmBPmPeYQ&ab_channel=Computerphile

- https://www.ted.com/talks/blaise_aguera_y_arcas_how_computers_are_learning_to_be_creative?language=en#t-898371
