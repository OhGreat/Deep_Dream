import numpy as np
import tensorflow as tf
import pathlib
import os
import PIL
import matplotlib.pyplot as plt
import IPython.display as display


# Check for corrupted files in directory
def get_corrupted_files(data_dir, verbose=0, ret_split=0):
    """
    Checks for corrupted files in a directory

    data_dir: directory of the data as pathlib.Path object
    verbose: print out corrupted file paths and number of total corrupt images
    ret_split: return lists of verified and corrupted images
    """

    count_opened = 0
    count_error = 0
    good_file_urls = []
    bad_file_urls = []

    # Iterate through files in directory
    for filename in os.listdir(data_dir):
        # File extension control
        if (filename.endswith('.jpg') or filename.endswith('.png') or 
            filename.endswith('.bmp') or filename.endswith('.JPG') or 
            filename.endswith('.jpeg')):
        
            curr_img_path = str(data_dir.resolve()) +"/"+filename
            # try to open the image
            try:
                img = PIL.Image.open(curr_img_path) # open the image file
                img.verify() # verify that it is an image
                good_file_urls.append(curr_img_path) #add to 'good' files
                count_opened += 1

            # exception for corrupted file
            except (IOError, SyntaxError) as e:
                bad_file_urls.append(curr_img_path)
                count_error += 1
                if verbose > 1:
                    print('error:', e)
                    print('Bad file:', filename) # print out the names of corrupt files
                

    if verbose > 0:
        print(f"Checked: {count_opened+count_error}, verified: {count_opened}, corrupted: {count_error}")
    if ret_split > 0:
        return good_file_urls, bad_file_urls




# Transform image to numpy array
def img_to_np(file_url, max_dim=None):
    """
    transform image to numpy array with PIL.

    file_url: url of the image we want to transform
    max_dim: can be used to set the maximum dimensions to transform the image
    """
    try:
        img = PIL.Image.open(file_url)
        if max_dim:
            img.thumbnail((max_dim, max_dim))
        img_np = np.array(img)
        return img_np

    except (IOError, SyntaxError) as e:
        print("bad file: ", file_url)


# MinMax normalization
def normalize_values(np_img):
    """
    Normalizes the values of an image as numpy array to values between 0 & 1 
    """
    x_min = np_img.min()
    x_max = np_img.max()
    x_norm = (np_img - x_min) / (x_max - x_min)
    return x_norm

def deprocess(img):
  """
  Deprocess an image preprocessed with InceptionV3 model
  """
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)


def show_np_img(np_img, use_pil=False):
    
    if use_pil:
         display.display(PIL.Image.fromarray(np.array(np_img)))
    else:
        plt.imshow(np_img)


def save_image(image, filename, dir=''):
    """
    Convert a numpy array to jpeg and save to directory.

    image: numpy array containing the image
    filename: the name of the file to save
    dir: the directory where we want to save the file
    """

    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Convert to bytes.
    image = image.astype(np.uint8)
    
    # Write the image-file in jpeg-format.
    with open(dir+filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')