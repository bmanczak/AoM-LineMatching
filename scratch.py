import argparse
import os
from matchingObjectsClass import matchingObjects, get_non_zero_objects



img_paths = sorted(os.listdir(frames_directory))
img_paths = np.random.choice(img_paths, size = int(0.4*len(img_paths)), replace = False) # take a random sample of 10$ of one directory

img_objects = np.array([matchingObjects(os.path.join(img_directory, path)) for path in img_paths])

# Apply hough transform to all objects in the array above
for obj in img_objects:
    obj.hough_lines()

non_zero_objects = get_non_zero_objects(img_objects)

### CREATE DATA FRAME
for obj in non_zero_objects:
    obj.rank_and_pick_lines(delta_angle=3, max_lines=7)  # apply line filtering

paths = []
angles = []
lines = []
lengths = []

for obj in non_zero_objects:
    for line_num in range(len(obj.lines)):
        paths.append(obj.path)
        lines.append(obj.lines[line_num])
        angles.append(obj.angle[line_num][0])
        lengths.append(obj.length[line_num][0])

data = {"path": paths, "angle": angles, "line": lines, "length": lengths}
data = pd.DataFrame(data)
data = data.sort_values(by=['angle', "length"]).set_index("path")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames-directory",
                        required = False,
                        type = str,
                        default = "frames/AV0539",
                        help = "Directory with the frames from which to extract the lines")

    parser.add_argument("--img-directory",
                        required=False,
                        type=str,
                        default="frames/contemporary/IMG_2864.jpeg",
                        help="Directory with the frames from which to extract the lines")