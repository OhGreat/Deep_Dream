import argparse
import tensorflow as tf
from classes.TiledGradients import *
from images_aux import *

def get_usable_layers(model):
        """
        Returns a list of all the usuable layers that we can feed as outputs for our model, 
        together with the list of names
        """
        usable_layers = []
        layer_names = []

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                usable_layers.append(layer)
                layer_names.append(layer.name)
        return usable_layers, layer_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", action="store", dest="img", type=str,
                        default="input/elephant.jpg")
    parser.add_argument("-img_max_dim", action="store", dest="img_max_dim",
                        help="Sets the maximum size of the image",
                        default=1024, type=int)
    parser.add_argument("-img_out", action="store", dest="img_out", type=str,
                        default="output")     
    parser.add_argument("-ts", action="store", dest="tile_size",
                        help="Sets the size of the tile for the random roll",
                        default=512, type=int)          
    parser.add_argument("-ss", action="store", dest="step_size", 
                        help="Sets the step_size per iteration",
                        default=0.01, type=float)
    parser.add_argument("-or", action="store", dest="octaves_range", 
                        nargs="+", help="Set octave range",
                        default=[-2,4], type=int)
    parser.add_argument("-os", action="store", dest="steps_per_octave",
                        help="Sets the number of steps per octave",
                        default=100, type=int)
    parser.add_argument("-osc", action="store", dest="octave_scale", 
                        help="Set scaling for each octave step",
                        default=1.2, type=float)
    parser.add_argument("-m", action="store", dest="model", 
                        help="Set the model to use",
                        default="inceptionResNet", type=str)
    parser.add_argument("-ml", action="store", dest="model_layers", 
                        nargs="+", help="Set the range of layers to consider",
                        default=None, type=int)
    args = parser.parse_args()
    print("arguments passed:",args)

    # Preprocess input image
    original_img = img_to_np(args.img,max_dim=args.img_max_dim)
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
    
    # Pick model usable layers
    layers, _ = get_usable_layers(base_model)
    layer_outputs = [layer.output for layer in layers]
    # Choose layers if they have not been selected
    # or if they have been selected incorrectly
    if  (args.model_layers == None) or (
        args.model_layers[0] > args.model_layers[1]) or (
        args.model_layers[1] >= len(layer_outputs)):
        print(f"Please choose starting and ending layer indexes, between 0 and {len(layer_outputs)}:" )
        start_idx,end_idx=map(int,input().split())
    # Else use the existing parameters
    else: start_idx, end_idx = args.model_layers[0], args.model_layers[1]
    print(f"Chosen layers: {start_idx} - {end_idx}")
    layer_outputs = layer_outputs[start_idx:end_idx]

    # Create model from selected layers
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layer_outputs)
    tiled_gradients = TiledGradients(dream_model)
        
    # Run deep dream algorithm
    img = tiled_gradients.run_deep_dream_with_octaves(  img=img, tile_size=args.tile_size, 
                                                        steps_per_octave=args.steps_per_octave, 
                                                        step_size=args.step_size, 
                                                        octaves=range(args.octaves_range[0], args.octaves_range[1]), 
                                                        octave_scale=args.octave_scale)

    # Deprocess and save final image
    img = tf.image.resize(img, original_img.shape[:2])
    img = tf.image.convert_image_dtype(img/255, dtype=tf.uint8)
    save_image(img, f"output/{args.img_out}.jpeg")

if __name__ == "__main__":
    main()