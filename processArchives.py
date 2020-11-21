import argparse
import os
from processingMethods import matchingObjects, get_non_zero_objects
import time
import pickle
import numpy as np
import pandas as pd

"""This script generates and saves a pickle of the pandas DataFrame and used for matching to a query image later"""

def print_config():
    """
    Prints all entries in the config variable.
    """
    print("[INFO]: Processing archives with following parameters ...")
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

def process_archives(config):
    """This script generates and saves a pickle of the pandas DataFrame and used for matching to a query image later"""

    # Get the image paths
    img_directory = os.listdir(config.frames_directory)
    img_paths = []
    temp_list = []
    for directory in img_directory:
        temp_list += list(os.listdir(os.path.join(config.frames_directory, directory)))

        for path in temp_list:
            img_paths.append(os.path.join(config.frames_directory, directory, path))

        temp_list = []

    img_paths = list(sorted(img_paths))

    np.random.seed(1998) # set seed for reproducability
    start = time.time()
    img_paths = np.random.choice(img_paths, size = int(config.sample*len(img_paths)), replace = False).astype(str) # take a random sample of the img_paths

    img_objects = np.array([matchingObjects(path, scale=config.scale) for path in img_paths])

    # Apply hough transform to all objects in the array above
    for obj in img_objects:
        obj.hough_lines()

    non_zero_objects = get_non_zero_objects(img_objects)

    for obj in non_zero_objects:
        obj.rank_and_pick_lines(delta_angle=3, max_lines=7)  # apply line filtering


    print("[INFO]: Running hough lines and filtering for {} images took {} seconds.".format(len(img_paths), round(time.time() - start, 2)))


    # Create a dataframe

    # INIT THE LISTS FOR THE PARAMETERS OF THE MATCHED LINES
    paths = []
    angles = []
    lines = []
    lengths = []
    scales = []

    non_zero_objects_dic = {}

    for obj in non_zero_objects:
        non_zero_objects_dic[obj.path] = obj
        for line_num in range(len(obj.lines)):
            paths.append(obj.path)
            scales.append(obj.scale)
            lines.append(obj.lines[line_num])
            angles.append(np.round(obj.angle[line_num][0], 2))
            lengths.append(np.round(obj.length[line_num][0], 2))

    data = {"path": paths, "scale": scales, "angle": angles, "line": lines, "length": lengths}
    data = pd.DataFrame(data)
    data = data.sort_values(by=['angle', "length"]).reset_index() # .set_index("path")

    # SAVE THE OBJECTS
    data.to_pickle("data/data.pkl")
    file_to_write = open("data/non_zero_object_dic.pickle", "wb")
    pickle.dump(non_zero_objects_dic, file_to_write)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames_directory",
                        required = False,
                        type = str,
                        default = "archives",
                        help = "Diectory to a folder that contains folders (e.g.AV...) with frames from videos ")

    parser.add_argument("--scale",
                        required=False,
                        type=float,
                        default=1.0,
                        help="Scale with which to process to process the images")

    parser.add_argument("--sample",
                        required=False,
                        type=float,
                        default=0.75,
                        help="The proportion of frames to be processed. Due to correlation of frames it is redundant to take all the frames")

    parser.add_argument("--delta_angle",
                        required=False,
                        type=int,
                        default=3,
                        help="The range within which lines are considered to be similiar for the sake of filtering similiar lines.")

    parser.add_argument("--max_lines",
                        required=False,
                        type=int,
                        default=7,
                        help="The maximum number of lines to be preserved in an image after filetring.")

    config = parser.parse_args()
    print_config()
    process_archives(config)
    print("[INFO]: Processing successful!")