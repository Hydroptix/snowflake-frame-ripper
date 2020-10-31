# snowflake-frame-ripper

Find unique frames from videos for training models.

## Usage
 `python3 frame-ripper.py [-t threshold] video_folder_path`
 
### video_folder_path

The tool will find all .mp4 files (shallowly) in the given directory. The folder path must be the last argument in the command.

### -t threshold

The float value below which a frame will be counted as unique. For the dataset I originally used. The similarity value is given using the TM_COEFF_NORM metric in OpenCV2's template matching function.

For my dataset, I found the right ratio of "unique" to "non-unique" frames to with a threshold value of 0.17, and that's the default for this program. Your mileage may vary, especially since OpenCV's template matching really can't tell when an image is being rotated. The template matching function gives tons of digits of precision, so be as precise as you like.