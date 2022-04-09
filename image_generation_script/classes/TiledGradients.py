import tensorflow as tf
from images_aux import *

class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[2], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)
        
        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                # This needs gradients relative to `img_rolled`.
                # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[y:y+tile_size, x:x+tile_size]
                    loss = self.calc_loss(img_tile)
                    print("loss:",loss)
                    print("img rolled:", img_rolled.shape)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0,1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 
        return gradients 

    def calc_loss(self, img):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = self.model(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        losses = []
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)

        return  tf.reduce_sum(losses)

    def run_deep_dream_with_octaves(self, img, steps_per_octave=20, step_size=0.01, 
                                octaves=range(-3,3), octave_scale=1.2):
        base_shape = tf.shape(img)
        initial_shape = img.shape[:-1]
        img = tf.image.resize(img, initial_shape)
        for octave in octaves:
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32)*(octave_scale**octave)
            new_size = tf.cast(new_size, tf.int32)
            img = tf.image.resize(img, new_size)

            for step in range(steps_per_octave):
                gradients = self(img, new_size)
                img = img + gradients*step_size
                img = tf.clip_by_value(img, -1, 1)

                if step % 10 == 0:
                    display.clear_output(wait=True)
                    show_np_img(deprocess(img), use_pil=True)
                    print ("Octave {}, Step {}".format(octave, step))
            
        result = deprocess(img)
        return result