import sys
import numpy as np
import argparse
import pathlib
import tensorflow as tf
import IPython.display as display
from classes.TiledGradients import *
from images_aux import *

def get_usable_layers(self):
        """
        Returns a list of all the usuable layers that we can feed as outputs for our model, 
        together with the list of names
        """
        usable_layers = []
        layer_names = []

        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                usable_layers.append(layer)
                layer_names.append(layer.name)
        return usable_layers, layer_names

def main():
    """ to add in parser:
            - usable layers line 60
            - img_to_np line 69
            - random_roll size line 70
            
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", action="store", dest="img", type=str,
                        default="input/elephant.jpg")
    parser.add_argument("-ss", action="store", dest="step_size", 
                        help="Sets the step_size per iteration",
                        default=0.01, type=float)
    parser.add_argument("-or", action="store", dest="octaves_range", 
                        nargs="+", help="Set octave range",
                        default=[-2,4], type=float)
    parser.add_argument("-os", action="store", dest="steps_per_octave",
                        help="Sets the number of steps per octave",
                        default=100, type=int)
    parser.add_argument("-osc", action="store", dest="octave_scale", 
                        help="Set scaling for each octave step",
                        default=1.2, type=float)
    parser.add_argument("-m", action="store", dest="model", 
                        help="Set the model to use",
                        default="inceptionResNet", type=str)
    args = parser.parse_args()
    print("arguments passed:",args)

    # Preprocess input image
    original_img = img_to_np(args.img,max_dim=1024)
    print("input image shape:",original_img.shape)
    # Choose model and preprocess image accordingly
    if args.model == 'inceptionV3':
        img = tf.keras.applications.inception_v3.preprocess_input(original_img)
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    elif args.model == 'inceptionResNet':
        img = tf.keras.applications.inception_resnet_v2.preprocess_input(original_img)
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
    else:
        print('Please select a valid model.')
        exit()
    
    # Pick model layers
    layers, names = get_usable_layers(base_model)
    layer_outputs = [layer.output for layer in layers]
    print(f"Choose starting and ending layer indexes, between 0 and {len(layer_outputs)}:" )
    start_idx,end_idx=map(int,input().split())
    print(f"Chosen layers: {start_idx} - {end_idx}")
    layer_outputs = layer_outputs[start_idx:end_idx]

    # Create model from selected layers
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layer_outputs)
    tiled_gradients = TiledGradients(dream_model)
        
    # Run deep dream algorithm
    img = tiled_gradients.run_deep_dream_with_octaves(img=img, steps_per_octave=args.steps_per_octave, step_size=args.step_size, 
                                octaves=range(1,3), octave_scale=args.octave_scale)

    #display.clear_output(wait=True)
    img = tf.image.resize(img, original_img.shape[:2])
    img = tf.image.convert_image_dtype(img/255, dtype=tf.uint8)

    save_image(img, f"output/output.jpeg")

if __name__ == "__main__":
    main()