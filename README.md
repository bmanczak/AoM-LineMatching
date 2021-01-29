# Artifacts Of Memory - Line Matching system: 
### extracting, filtering and matching lines from a set of frames to a query image .

Welcome to this repo! In this repo you can find an efficient implementation of detecting, filtering and matching the lines from a set of frames to the lines detected in the query image.
You can find step by step details in ``walkthrough.ipynb``. This project has been realized in collaboration with the *Netherlands Institute of Sound and Vision* and *Superposition* in the context of their project [**Artifacts of Memory**](https://superposition-cc.medium.com).  We used the frames extracted from video archives as the extraction set. 

### The pipeline
The pipeline consists of two main parts. The first part concerns dealing with the video archives and the second part with matching the lines from a query image with the matched lines from the archive.


#### **Step 1**: Preprocess the video archives into a format from which we can extract lines. This has to only be done once.
- Sample frames from videos to work on static data
- For each frame, run a line extraction together with additional line features such as line angle and length.
- Discard the frames with no detected lines and filter out weak (short) and closely similiar lines (reduce noise and dimensionality for later retrival)
- Save the information about the filtered data in an format that is easy to query (we chose Pandas dataframe)

This step can be applied with ``processArchives.py`` script (customize with command line arguments). This script uses methods in ``processingMethods.py``.
#### **Step 2**: Extract and match the lines from the query image to the database of candidate matches from the archive.
- run the line extraction and filter algorithm on the query image
- choose the line(s) to be matched. How? We prefer long, non-horizontal lines.
- perform the matching procedure
- fetch the match and overlay it on the query image

This step can be applied with ``overlay.py`` script (customize with command line arguments). This script uses methods from ``matchingMethods.py``.

### Environment 
We provide a conda environment called ``AOM`` which contains all packages you need to execute the scripts in this repo.

### Structure
```
ArtifactsOfMemory
│   walkthrough.ipynb
│   processingMethods.py
│   processArchives.py
│   matchingMethods.py
│   overlay.py
│   
│   archives/ % contains at least one folder
│      folderWithImgs/  % folder containing images from which to extract line
│   
│   frames/ 
│       contemporary/ % folder containing the query images on which the matches should be overlayed
│       outputs/ % Directory where the overlayed images should be stored, 
│
│   data/ % stores the data produced by processArchives.py
```
Note that folders ``archives``, ``frames`` and ``data`` is not in this repo. Prior to execution you must **create these folders yourself**.
### Example execution
```
cd ../ArtifactsOfMemory
conda activate AOM
python processArchives.py --scale 0.8 --sample 0.5 
python overlay.py --num_lines 2 --hough_params 200,150,25
```
### Example output
![Example output](https://github.com/blazejmanczak/ArtifactsOfMemory/blob/master/example_out.png)
