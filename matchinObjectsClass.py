import cv2
import matplotlib.pyplot as plt
import numpy as np


# Set the parameters

rho_resolution = 1
theta_resolution = np.pi / 180
threshold = 100  # The minimum number of intersections in hough space to "detect" a line. higher -> less lines

# Additional parameters for probabilistic
minLineLength = 150  # The minimum number of points that can form a line. Lines with less than this number of points are disregarded. higher -> less lines
maxLineGap = 30  # The maximum gap between two points to be considered in the same line. higher -> more lines


def apply_hough(img, minLineLength=minLineLength, maxLineGap=maxLineGap, probabilistic=True):
    """Applies the Hough transform"""

    if probabilistic:
        edges_image = cv2.Canny(img, 50, 200, None, apertureSize=3)  # without Blur
        hough_lines = cv2.HoughLinesP(edges_image, rho_resolution, theta_resolution, threshold, None, minLineLength,
                                      maxLineGap)

    else:
        blurred_image = cv2.GaussianBlur(img, (7, 7), 0)  # filter out week lines
        edgs_image = cv2.Canny(blurred_image, 50, 150, apertureSize=3)  # 120 is another visually optimal value
        hough_lines = cv2.HoughLines(edges_image, rho_resolution, theta_resolution, threshold)

    return hough_lines


def get_angle(hough_lines, radians=False):
    if radians:
        return np.arctan2(hough_lines[:, :, 3] - hough_lines[:, :, 1], hough_lines[:, :, 2] - hough_lines[:, :, 0])
    else:
        return np.arctan2(hough_lines[:, :, 3] - hough_lines[:, :, 1],
                          hough_lines[:, :, 2] - hough_lines[:, :, 0]) * 180 / np.pi  # better for numerical stability


def plot_lines(img, hough_lines, probabilistic=True):
    """Plots only hough lines"""

    original_image_with_hough_lines = img.copy()
    cmap = "gray" if img.shape[-1] == 3 else None

    if probabilistic:
        for i in range(0, len(hough_lines)):
            l = hough_lines[i][0]
            cv2.line(original_image_with_hough_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
            # break

    else:
        hough_lines_image = np.zeros_like(img)
        draw_lines(hough_lines_image, hough_lines)
        original_image_with_hough_lines = weighted_img(hough_lines_image, img)

    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(original_image_with_hough_lines, cmap=cmap)


"""
# Example usage
probabilistic = True

#img = plt.imread("frames/contemporary/IMG_2864.jpeg")
hough_lines = apply_hough(img, probabilistic = probabilistic)
plot_lines(img, hough_lines, probabilistic = probabilistic)
"""


class matchingObjects:

    def __init__(self, path=None, margin=50, img=None):

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

    def hough_lines(self, probabilistic=True, radians=False):
        self.lines = apply_hough(self.img, probabilistic=probabilistic)

        if self.lines is not None:  # if hough found something
            self.angle = get_angle(self.lines, radians=radians)  # radians more stable

            x_diff = self.lines[:, :, 2] - self.lines[:, :, 0]  # if 0 then slope will be -inf -> vertical line
            y_diff = self.lines[:, :, 3] - self.lines[:, :, 1]
            self.slope = y_diff / x_diff
            self.length = np.sqrt(x_diff ** 2 + y_diff ** 2)

    def rank_and_pick_lines(self, delta_angle=1, max_lines=None):
        # print("Intiail max:", np.max(self.length))
        initial_max = np.max(self.length)
        # print("initial", intial_max)
        if self.lines is not None:
            lst0 = self.lines
            order = np.arange(0, len(lst0)).reshape(-1, 1)
            lst1 = self.angle
            lst2 = self.length
            merged = np.concatenate([lst1, lst2, order], axis=1)
            new_order = np.lexsort((lst2, lst1), axis=0)  # sorts first by angle then by length
            merged_new = merged[new_order]
            # print("merged_new:", merged_new)

            mask = (np.diff(merged[new_order], axis=0)[:, :, 0] < delta_angle)
            series = False

            for i in range(len(mask)):  # marks
                if mask[i] == True:
                    series = True
                elif (mask[i] == False) and (series == True):
                    mask[i] = True  # make up for the offset in the mask
                    series = False  # break the series

            grouping_mask = np.concatenate((mask, np.array([[False]])))
            accum = []
            temp = []

            for i in range(len(grouping_mask)):
                # print("current row", merged_new[i,:,:])
                if grouping_mask[i] == False:
                    if (len(temp) > 0):
                        # print("temp:", temp)

                        accum.append(np.array(temp)[np.argmax(np.array(temp), axis=0)[0][1]])
                        # print("added to accum:", np.array(temp)[np.argmax(np.array(temp), axis = 0)[0][1]])
                        temp = []
                    accum.append(merged_new[i, :, :])


                else:  # if grouping_mask[i] == True:

                    if len(temp) > 0:
                        if abs(merged_new[i, :, :][0][0] - temp[-1][0][0]) < delta_angle:
                            temp.append(merged_new[i, :, :])
                        else:
                            # print("else temp:", temp)
                            accum.append(np.array(temp)[np.argmax(np.array(temp), axis=0)[0][1]])
                            # print("added to accum:", np.array(temp)[np.argmax(np.array(temp), axis = 0)[0][1]])
                            temp = []
                            temp.append(merged_new[i, :, :])
                    else:

                        temp.append(merged_new[i, :, :])
            # print("pre sort accum", accum)

            accum = np.array(accum)
            accum = accum[np.argsort(accum[:, :, 1], axis=0)]  # sort by length
            # print("Accum", accum)
            if max_lines is not None:  # if the maximum number of lines to be returned is specifed, pick the longest lines
                accum = accum[-max_lines:]
            # print(accum)
            cleaned_order = list(accum[:, :, :, 2].flatten().astype(int))
            # print("Cleaned orderd:", cleaned_order)
            # print("self lines before slicing", [self.lines, self.length])

            self.lines = self.lines[cleaned_order]
            self.angle = self.angle[cleaned_order]
            self.length = self.length[cleaned_order]
            # print("self lines AFTER slicing", [self.lines, self.length])
            self.slope = self.slope[cleaned_order]
            final_max = np.max(self.length)
            # print(final_max, intial_max)
            assert (abs(final_max - initial_max) < 0.01)  # making sure
            # print(np.max(self.length))

        def plot_matches(self, matches, buffer=5):
            self.matched_img = np.copy(self.img)
            pass


def get_non_zero_objects(obj_list):
    new_obj_list = []
    count_lines = 0
    for obj in obj_list:
        if obj.lines is not None:
            new_obj_list.append(obj)
            count_lines += len(obj.lines)

    print("[INFO]: {}% of input list contain lines".format(round(len(new_obj_list) / len(obj_list) * 100), 2))
    print("[INFO]:Given that the img contains a line, on average there are {} detected lines per image".format(
        round(count_lines / len(new_obj_list), 2)))
    return new_obj_list


