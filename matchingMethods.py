from processingMethods import matchingObjects
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def match_lines(data, obj, delta_angle=3):
    """

    Matches the lines of the given MatchingObject against the lines detected and stored in data

    Parameters:
    ------------------------
    data: pandas DataFrame
        database of lines with columns: path, angle, line, length
    obj: object of class MatchingObject
        an object of class matchingObjects for the query image
    delta_angle: int
        what is the furthest angle to yield a match? Note that the matches are first filtered on angle and only then on length
    TO-DO: length_prop - the proportion of length to be qualified as a match, e.g. length_prop=0.5 means that the matched length must be at least

    Returns: pandas DataFrame
        dataframe of lines with columns: path, scale, angle, line, length, line coords in the input image

    """
    matches = pd.DataFrame()
    for i in range(len(obj.lines)):
        close_angle_data = data.loc[(data["angle"] < obj.angle[i][0] + delta_angle) & (data["angle"] > obj.angle[i][
            0] - delta_angle)]  # we first sort by angle as it is more important to match than length
        match = close_angle_data.iloc[(close_angle_data['length'] - obj.length[i][0]).abs().argsort()][
                :1]  # given the angle data, find the closest match in length
        match["obj_line"] = [obj.lines[i]]
        match["obj_angle"] = obj.angle[i]  # just to show how great the match is, can be deleted later
        match["obj_length"] = obj.length[i]
        matches = pd.concat([matches, match])
    matches.dropna(inplace=True)
    matches["line"] = matches["line"].apply(lambda x: x.flatten())
    matches["obj_line"] = matches["obj_line"].apply(lambda x: x.flatten())

    return matches


def score_the_line(matches, normalizing_stats = [71.73, 26.70, 254.71, 94.19],
                   angle_weight=1.0, length_weight=1.0, num_lines=None):
    """
    Scores the lines striking a balance between angle (vertical lines preferred) and length (long lines preferred).
    The scoring method is based on a composite score of (normalized) angle and length, with user-specified weight for each. The normalization of each
    is done by subtracting the mean and dividing by the std.

    Parameters:
    -----------------------
    matches: pandas DataFrame
        contains information about the matched lines as returned by match_lines
    normalizing_stats: list like
        a list-like object contating mean and standard deviation of the absolute values of angle and length respectively (list of length 4).
        Defeault contains means and avergaes from 5 videos from testing phase.
    angle_weight: float
        weight of the angle in the score calculation
    length_weight: float
        weight of the length in the score calculation
    data: pandas DataFrame
        database of lines from the archive with columns: path, angle, line, length. Used to calculate the statistics if not supplied in normalizing_stats
    num_lines: int
        how many lines to return. Takes num_lines with the highest score


    Returns:
    ----------------------
        dataframe of lines with columns: path, scale, angle, line, length, line coords in the input image, score
    """
    matches_normalized = matches.copy()
    angle_mean, angle_std, length_mean, length_std = normalizing_stats

    matches_normalized = matches.copy()

    matches_normalized["angle_normalized"] = (abs(matches_normalized["angle"]) - angle_mean) / angle_std
    matches_normalized["length_normalized"] = (matches_normalized["length"] - length_mean) / length_std
    matches_normalized["score"] = angle_weight * matches_normalized["angle_normalized"] + length_weight * \
                                  matches_normalized["length_normalized"]

    if num_lines == None:
        best_matches = matches_normalized.sort_values(by="score", ascending=False)
    else:
        best_matches = matches_normalized.sort_values(by="score", ascending=False)[:num_lines]
    return best_matches


def sample_line(matches, num_lines=1, factor=2):
    """
    Samples num_line lines from 'matches' using random sampling where each consecutive row is 'factor'
    times less likely to be selected.

    Parameters:
    -------------------------------
    matches: pandas DataFrame
        dataframe with lines ordered by score
    num_lines: int
        how many lines to return. Takes num_lines with the highest score.
        If num_lines bigger than matches, all matches are returned.
    factor: float
        defines how the odds of each consecutive rows change

    Returns:
    ----------------------------------------------

    matches_new.sample(...): padnas data frame
        Samples of matches

    """
    matches_new = matches.copy()
    start_val = 1  # does not reallt matter as numbers will be
    probs = [start_val]
    for i in range(matches.shape[0] - 1):
        probs.append(probs[i] / factor)

    probs = np.array(probs)
    probs = probs / np.sum(probs)
    matches_new['probs'] = probs

    return matches_new.sample(n=min(num_lines, matches.shape[0]), replace=False, weights='probs', axis=0)


def get_the_line_rect(line_coords, img, margin_x=0, margin_y=0):
    """
    Returns the rectangluar crop with diganol being the line with specified 'line_coords' optionally modified by a margin.

    Parameters:
    ---------------------------------------
    line_coords: list
        Has format  [x_start,y_start, x_end,y_end]
    img: np.array
        an imgae
    margin_x: int
        How should the line be modified in x direction
    margin_y: int
        How should the line be modified in x direction

    Returns:
    ---------------------------------------
    matched_rect: np.array
        the cropped rectangle
    final_coords: list
        coordinates of the line the input image taking into account the margin and img_size

    """
    x_s, y_s, x_e, y_e = line_coords  # get the coordinates of the line
    height_match, width_match, _ = img.shape
    x_s, y_s, x_e, y_e = max(min(x_s - margin_x, x_e - margin_x), 0), max(min(y_s - margin_y, y_e - margin_y), 0), min(
        max(x_s + margin_x, x_e + margin_x), width_match), min(max(y_s + margin_y, y_e + margin_y), height_match)

    matched_rect = img[y_s:y_e, x_s:x_e]
    final_coords = [x_s, y_s, x_e, y_e]
    return matched_rect, final_coords


def overlay_on_img(img, matches, non_zero_objects_dic, margin_x=0, margin_y=0, adaptive_margin=True):
    """
    Takes the img and overlays the match(es) from 'matches' on it.

    Parameters:
    ---------------------------------------
    img: np.array
        an imgae
    matches: pandas DataFrame

    margin_x: int
        How should the line be modified in x direction
    margin_y: int
        How should the line be modified in x direction
    adaptive_margin: bool
        Specifies if the automatic margin should be performed.
        Automatic margin adds the margin based on trainge weight function. The added margin is 0 at 45 degrees and symmetric around 45 degrees.


    Returns:
    ---------------------------------------
    matched_rect: np.array
        the cropped rectangle

    """
    new_img = img.copy()

    if adaptive_margin == True:
        weights = np.concatenate((np.linspace(start=30, stop=0, num=45),
                                  np.linspace(start=0, stop=30, num=46)))  # maximally 15 pixels will be added
        weights_dic = {}
        for angle, weight in enumerate(weights):
            weights_dic[angle] = round(weight)

    for row_num in range(matches.shape[0]):
        row = matches.iloc[row_num]  #
        the_match_obj = non_zero_objects_dic[row["path"]]  # fetch the object from dic

        if adaptive_margin:
            angle = round(abs(row["angle"]))
            if angle < 45:  # increase the margin in y direction
                margin_y = weights_dic[angle]
            else:
                margin_x = weights_dic[angle]

        matched_rect, _ = get_the_line_rect(line_coords=row['line'], img=the_match_obj.img, margin_x=margin_x,
                                            margin_y=margin_y)  # get the part to paste

        input_img_rect, input_coords = get_the_line_rect(line_coords=row['obj_line'], img=img, margin_x=margin_x,
                                                         margin_y=margin_y)
        height_input, width_input, _ = input_img_rect.shape

        matched_rect = cv2.resize(matched_rect, (width_input, height_input))

        new_img[input_coords[1]:input_coords[3], input_coords[0]:input_coords[2]] = matched_rect

    return new_img


def all_in_one(path, data, non_zero_objects_dic,num_lines = 1, normalizing_stats=[71.73, 26.70, 254.71, 94.19],
               params_hough={"threshold": 200, "minLineLength": 150, "maxLineGap": 25}):
    """
    Given a query image and database of matches, finds a match and overlays it on the query image.

    Parameters:
    ---------------------------------------
    path: str
        A path to an image. Can be modified for obj to be read directly from array, see matchingObjects init
    data: pandas DataFrame
        database of lines with columns: path, angle, line, lengtg
    non_zero_objects_dic: dictionary
        Dictionary with keys being paths to the images in the archive that have lines and keys being the matchingObjects that store them. Allows for fast overlaying.
    num_lines: int
        how many matches to overlay. If num_lines bigger than matches, all matches are overlayed..
    normalizing_stats: list like
        a list-like object contating mean and standard deviation of the absolute values of angle and length respectively (list of length 4).
        Defeault contains means and avergaes from 5 videos from testing phase.
    params_hough: dictionary
        dictionary storing parameter values for hough transform

    Returns:
    --------------------------------------
    overlayed: np.array
        An image with overlayed match

    """

    img = plt.imread(path)  # read in query image
    obj = matchingObjects(img=img, scale=1, margin=0)  # make it a matchingObject class
    obj.hough_lines(radians=False, **params_hough)  # detect lines
    obj.rank_and_pick_lines(delta_angle=3, max_lines=None)  # filter similiar lines
    matches = match_lines(data, obj)  # find matches

    # Calculate the score for each candidate match
    matches = score_the_line(matches, normalizing_stats, num_lines=None)
    overlayed = overlay_on_img(img, sample_line(matches, num_lines=num_lines), non_zero_objects_dic,
                               adaptive_margin=True)  # randomly sample num_lines lines and overlay on image

    return overlayed