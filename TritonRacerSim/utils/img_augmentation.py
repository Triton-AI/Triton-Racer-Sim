"""
Augment image data with car-like squares 
to improve neglegence of the neural network to rival cars

The goal is to train a neural network that would not be influenced by rival cars
That are not present at training

Other parts of the program should be looking out for rival cars
"""



spawn_limit_ratio: 0.6 # ratio of the frame below which a car image will spawn.
                       # e.g. on a 400 pixel high image with this value set to 0.6, 
                       # a car image will spawn in the bottom 240 pixel height range
                       # So that we don't have car image in the sky

spawn_size_ratio_min: 0.2 # ratio of the width of the spawned image to the frame width
                          # at the spawn limit defined above
                          # e.g. on a 300 * 400 (h * w) image with spawn_limit_ratio = 0.5,
                          # a car image spawned at 150 pixel height will have a width of 80.

spawn_size_ratio_max: 0.5 # ratio of the width of the spawned image to the frame width
                           # at the bottom of the image
                           # e.g. on a 300 * 400 (h * w) image,
                           # a car image spawned at the bottom will have a width of 200.


def get_img_path(idx):
    return f'img_{idx}.jpg'

