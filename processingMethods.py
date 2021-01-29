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

import cv2
import matplotlib.pyplot as plt
import numpy as np

def mask_clusters(arr, delta_angle):
    """
    Createas a boolean array where clusters are interuptted with an if statement
    
    Parameters:
    ------------------------
    arr: np array
        contains sorted (angle) values, shape [#num_lines, 1]
    delta_angle: int
        what is the furthest angle to yield a match? Note that the matches are first filtered on angle and only then on length
   
    Returns:
    bool_arr: pandas DataFrame
        a boolean array with the same shape as arr
    
    """
    arr = list(arr) # get just the numbers of the array in the list
    bool_arr = []
    
    start_cluster_val = arr[0] # first element starts the cluster
    
    for cluster_element in arr:
        if abs(cluster_element - start_cluster_val) <= delta_angle:
            bool_arr.append(True)
        else:
            bool_arr.append(False)
            start_cluster_val = cluster_element
            
    bol_arr = np.array(bool_arr).reshape(-1,1)
    return bol_arr

    
def apply_hough(img, threshold=100, minLineLength=150, maxLineGap=30):
    """
    Applies the probabilistic Hough transform on the given img with given parameters.
    Parameters have to be adjusted for the resolution of the image and charactertistics of the input image.
    The high-res images are recommened to be scaled down to no more than 1500 pixels on the longest dimension as the transform doesn not perform well for higher res images

    Example parameters that proved to be working in different settings:

    params_low_res_hough_prob = {"threshold": 100, "minLineLength":150, "maxLineGap":30} # use in case large images (more than 1000 pixels in width/heigh) # for images around 500 pixels width/lentgh
    params_high_res_hough_prob = {"threshold": 200, "minLineLength":150, "maxLineGap":25} # use in case large images (more than 1000 pixels in width/heigh) # for images around 1000 pixels width/length

    Parameters:
    ----------------------------
    img: np.array
        img in the form of a numpy array
    threshold: int
        The minimum number of intersections in hough space to "detect" a line. higher -> less lines
    minLineLength: int
        the minimum number of points that can form a line. Lines with less than this number of points are disregarded. higher -> less lines
    maxLineGap: int
        The maximum gap between two points to be considered in the same line. higher -> more lines

    Returns:
        hough_lines: np.array
            A 3 dimensional array of format [num_lines, 1, 4]. The last dimension stroes x_start, y_start , x_end, y_end of a line.

    """
    rho_resolution = 1  # the discretization parameter for rho: not recommended to change
    theta_resolution = np.pi / 180  # the discretization parameter for theta: not recommended to change

    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)  # filter out week lines. Bigger filters apply more blure -> less lines
    edges_image = cv2.Canny(blurred_image, 50, 120, None, apertureSize=3)  # applying Canny on the blurred image.
    hough_lines = cv2.HoughLinesP(edges_image, rho_resolution, theta_resolution, threshold, None, minLineLength,
                                  maxLineGap)

    return hough_lines


def get_angle(hough_lines, radians=False):
    """
    Calculates the angle of each line wrt. to the image plain.

    Parameters:
    -------------------
    hough_lines: np.array
            A 3 dimensional array of format [num_lines, 1, 4]. The last dimension stroes x_start, y_start , x_end, y_end of a line.
    radians: bool
        Whether to use radians. If False, uses degrees (better for numerical stability).

    Returns:
    -------------------
    out: np.array
        Array of shape [num_lines, 1] storing the angle of each line. Note that the degrees fall in range [-90,90] degrees.
    """
    if radians:
        return np.arctan2(hough_lines[:, :, 3] - hough_lines[:, :, 1], hough_lines[:, :, 2] - hough_lines[:, :, 0])
    else:
        return np.arctan2(hough_lines[:, :, 3] - hough_lines[:, :, 1],
                          hough_lines[:, :, 2] - hough_lines[:, :, 0]) * 180 / np.pi  # better for numerical stability


def plot_lines(img, hough_lines):
    """
    Plots the hough_lines on the orginal image and displays it next to the orginal image.
    Used in Jupyter Notebook for inspection. Can be modified to save the figure.

    Parameters:
    -------------------
    img: np.array
        img in the form of a numpy array
    hough_lines: np.array
            A 3 dimensional array of format [num_lines, 1, 4]. The last dimension stroes x_start, y_start , x_end, y_end of a line.
    Returns: none
    """

    original_image_with_hough_lines = img.copy()
    cmap = "gray" if img.shape[-1] == 3 else None


    for i in range(0, len(hough_lines)):
        l = hough_lines[i][0]
        cv2.line(original_image_with_hough_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)


    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(original_image_with_hough_lines, cmap=cmap)



class matchingObjects:
    """
    Stores frequently accessed information about the images and contains methods useful for line fitting.

    Attributes:
    --------------------
    path: str
        if file is read from directory, path stores that directory
    img: np.array
        stores the image in the RGB fashion
    margin: int
        defines how much the borders should be clipped. Useful for old images that have black borders resulting in fake line detections
    shape: np.array
        stores shape of the image
    scale: float
        the scaling factor compared to the original image
    lines: np.array
        strores the x_start,y_start, x_end, y_end corrdinates of the line in the image with a given scale and margin (not the orignal image)
    angle: np.array
        stores the angle of the detcted lines corresponding to the lines at the same index as in the "lines" atrribute
    slope: np.array
        stores the slope of the detcted lines corresponding to the lines at the same index as in the "lines" atrribute. Not used -> commented out.
    length:
        stores the length of the detcted lines corresponding to the lines at the same index as in the "lines" atrribute


    Methods:
    -------------------

    hough_lines(self, radians = False, **kwargs):
        Applies probabilistic hough transform and calculates the characteristics of found lines

    rank_and_pick_lines(self, delta_angle = 1, max_lines = None):
        Filters out lines having similiar angles (taking the longest line out of the "similiar ones") to later limit the dimensionality of the database of lines.

    """

    def __init__(self, path=None, margin=50, img=None, scale=1):
        """
        Parameters:
        -----------------
        path: str
            if file is read from directory, path stores that directory. If supplied, img is expected to be none.
        margin: int
            defines how much the borders should be clipped. Useful for old images that have black borders resulting in fake line detections.
            Remeber to adjust your margin with scale!
        scale: float
            the scaling factor compared to the original image
        img: np.array
            stores the image in the RGB fashion. If supplied, path is expected to be none.
        """
        self.scale = scale
        if img is None:  # read in the image from path

            self.path = path

            if margin > 0:  # deals with black borders that result in many fake line detections
                self.img = plt.imread(self.path)[margin:-margin, margin:-margin]
            else:
                self.img = plt.imread(self.path)


        elif path is None:  # np array image is provided
            self.path = None
            if margin > 0:  # deals with black borders that result in many fake line detections
                self.img = img[margin:-margin, margin:-margin]
            else:
                self.img = img

        self.shape = self.img.shape

        if scale != 1:
            self.img = cv2.resize(self.img, (int(self.shape[0] * self.scale), int(self.shape[1] * self.scale)))
            self.shape = self.img.shape

    def hough_lines(self, radians=False, **kwargs):
        """
        Applies  probabilistic hough transform to find lines. Additionally, for each line, it determines the angle, slope and length.

        Parameters:
        -----------------
        radians: bool
            Determines whether to use radians. False is preffered due to possible numerical underflow problems later.
        **kwargs: dict, optional
            Additional arguments specyfing the parameters of the hough transform. Useful as the default has been optimized for archival, medium resolution photos.
        """
        self.lines = apply_hough(self.img, **kwargs)

        if self.lines is not None:  # if hough found something
            self.angle = get_angle(self.lines, radians= False)  # radians more stable

            x_diff = self.lines[:, :, 2] - self.lines[:, :, 0]  # if 0 then slope will be -inf -> vertical line
            y_diff = self.lines[:, :, 3] - self.lines[:, :, 1]
            # self.slope = y_diff/x_diff # can be calculated if needed.
            self.length = np.sqrt(x_diff ** 2 + y_diff ** 2)

    def rank_and_pick_lines(self, delta_angle = 1, max_lines = None):
        """
        Filters out lines having similiar angles (taking the longest line out of the "similiar ones") to later limit the dimensionality of the database of lines.
        
        Parameters:
        -----------------
        delta_angle: float
            defines how close the angles have to be considered 'similiar' in terms of the angle
        max_lines:int
            specifiecs how many lines should be kept after filtering. The longest max_lines number of lines are kept.
        """

        initial_max = np.max(self.length)
        if self.lines is not None:
            lst0 = self.lines
            order = np.arange(0, len(lst0)).reshape(-1,1)
            lst1 = self.angle
            lst2 = self.length
            merged = np.concatenate([lst1, lst2, order], axis = 1)
            new_order = np.lexsort((lst2, lst1), axis = 0) # sorts first by angle then by length
            merged_new = merged[new_order]  #
    
            
            grouping_mask = mask_clusters(merged[new_order][:,:,0], delta_angle) 
            accum = [] #stores the longest line from found clusters 
            temp = [] # empty list for booking within clusters of similiar lines

            #print(grouping_mask)
            for i in range(len(grouping_mask)): 
                if grouping_mask[i] == True:
                    temp.append(merged_new[i,:,:])
                else:
                    accum.append(np.array(temp)[np.argmax(np.array(temp), axis = 0)[0][1]])
                    temp = []
                    temp.append(merged_new[i,:,:])
            if len(temp)> 0: # push the last cluster 
                accum.append(np.array(temp)[np.argmax(np.array(temp), axis = 0)[0][1]])                    
    
            
            accum = np.array(accum)
            accum = accum[np.argsort(accum[:,:,1], axis = 0)] # sort by length
            #print("accum",accum)
            if max_lines is not None: # if the maximum number of lines to be returned is specifed, pick the longest max_lines lines
                accum = accum[-max_lines:]
            cleaned_order = list(accum[:,:,:,2].flatten().astype(int))
            
            
            #Update the attribute values
            self.lines = self.lines[cleaned_order]
            self.angle = self.angle[cleaned_order]
            self.length = self.length[cleaned_order]
            final_max = np.max(self.length)
        
            assert (abs(final_max - initial_max) < 0.01) # making sure the line of max length is preserved

        def plot_matches(self, matches, buffer=5):
            self.matched_img = np.copy(self.img)
            pass


def get_non_zero_objects(obj_list):
    """
    Filters out images where no lines were found.

    Parametrs:
    ------------------
    obj_list: list
        List of elements of class matchingObjects

    """
    new_obj_list = []
    count_lines = 0
    for obj in obj_list:
        if obj.lines is not None:
            new_obj_list.append(obj)
            count_lines += len(obj.lines)

    print("[INFO]: {}% of input list contain lines".format(round(len(new_obj_list) / len(obj_list) * 100), 2))
    print("[INFO]: Given that the img contains a line, on average there are {} detected lines per image".format(
        round(count_lines / len(new_obj_list), 2)))
    return new_obj_list
