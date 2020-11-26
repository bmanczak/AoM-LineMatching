# Copyright 2020-present, Netherlands Institute for Sound and Vision (Blazej Manczak)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from matchingMethods import all_in_one
import argparse
import os
import pandas as pd
from PIL import Image
import time

def print_config():
    """
    Prints all entries in config variable.
    """
    print("[INFO]: Overlaying with follwoing  parameters ...")
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

def overaly(config):
    """Performs the overlaying"""
    print("[INFO]: Loading in the pickeled data ... ")
    img_names = os.listdir(config.path_dir, )
    img_paths =[os.path.join(config.path_dir, name) for name in img_names]
    data = pd.read_pickle(config.data_directory)
    non_zero_objects_dic = pd.read_pickle(config.non_zero_objects_dic_directory)

    threshold, minLineLength, maxLineGap = [int(param) for param in config.hough_params.split(",")] # parse hough parameters

    start_time = time.time()
    count = 0

    for img_path in img_paths:
        try:
            img_array = all_in_one(path = img_path, data = data, non_zero_objects_dic = non_zero_objects_dic ,num_lines = config.num_lines, normalizing_stats=[71.73, 26.70, 254.71, 94.19],
                   params_hough={"threshold": threshold, "minLineLength": minLineLength, "maxLineGap": maxLineGap})
            im = Image.fromarray(img_array)
            im.save(os.path.join(config.save_dir ,"overlayed_" + img_path.split("/")[-1]))
            count += 1
        except Exception as e: print("Overlaying failed for path {} with exception {} ".format(img_path, e))
    end_time = time.time()

    print("[INFO]: overalying and saving took on average {} seconds per query image".format(round((end_time-start_time)/count,4)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_dir",
                        required=False,
                        type=str,
                        default = "frames/contemporary" ,
                        help="Directory containing images on which the matching should be done. All images in the directory will be matched. The directory should contain only images.")

    parser.add_argument("--save_dir",
                        required=False,
                        type=str,
                        default="frames/outputs",
                        help="Directory where the overlayed images should be stored.")

    parser.add_argument("--data_directory",
                        required = False,
                        type = str,
                        default = "data/data.pkl",
                        help = "Diectory to a pickle file of the processed archives")

    parser.add_argument("--non_zero_objects_dic_directory",
                        required=False,
                        type=str,
                        default="data/non_zero_object_dic.pickle",
                        help="Diectory to a pickle file of the processed matchingObjects that contain a line")

    parser.add_argument("--num_lines",
                        required=False,
                        type=int,
                        default=1,
                        help="How many lines should be overlayed? If num_lines bigger than matches, all matches are overlayed.")

    parser.add_argument("--hough_params",
                        required=False,
                        type=str,
                        default="200,150,25",
                        help="What parameters to use for line detection? Argument is expected to be a string of integers seperated by a comma. \
                             Consecutive ints stand for threshold, minLineLength and maxLineGap respectively.")


    config = parser.parse_args()
    print_config()
    overaly(config)
    print("[INFO]: Overlaying successful!")