"""
====================================
Filename:         orientation.py 
Author:              Joseph Farah 
Description:       Contains classes and methods for orientation of 
                    img files.
====================================
Notes
     
"""
#------------- imports -------------#
import os
import copy
import shutil
import numpy as np
import datetime as dt
from tqdm import tqdm



#------------- classes -------------#
class Point(object):
    """
    Object for manipulating and sequencing images taken on PPS route.
    """

    def __init__(self, timestamp, fpath: str, slate, im, dng: str):

        self.timestamp = timestamp
        """Timestamp corresponding to image."""
        self.fpath = fpath
        """Folderpath to JPG version of image."""
        self.dng = dng
        """Folderpath to DNG version of image."""
        self.slate = slate
        """Open or end slate or None."""
        self.im = im
        """Image contents."""
        self.tag = self.fpath.split('/')[-1].split('.')[0]
        """Tag containing information assigned by Friends of Pando."""
        self.timestamp_old = timestamp
        """Backup version of timestamp to preserve."""

    def __sub__(self, other):
        """
            Overwrite subtraction method to perform timestamp subtraction
        
            **Args**:
        
            * other (Point): additional Point object to perform delta.
        
            **Returns**:
        
            * delta (float): time delta between two Point objects in seconds.
        
        """
         
        
        diff = self.timestamp - other.timestamp
        return diff.seconds

    @staticmethod
    def get_tag(path: str):
        """
            Extract tag containing information assigned by Friends of Pando.
        
            **Args**:
        
            * path (str): filepath to image.
        
            **Returns**:
        
            * tag (str): tag of image.
        
        """
         
        
        return path.split('/')[-1].split('.')[0]


#------------- functions -------------#
def statistical_sequence(points, start_index, end_index, output_path):

    """
        Perform statistical sequencing sort of Point objects.
    
        **Args**:
    
        * points (list): list of Point objects to be sorted.
        * start_index (int): index of open slate.
        * end_index (int): index of end slate.
        * output_path (str): directory to save sorted images.
    
        **Returns**:
    
        * first_half_bin (list): first half of Point objects, sorted.
        * second_half_bin (list): second half of Point objects, sorted.
        * new_points (list): merged list of sorted Point objects.
        * diffs_combined_mean: statistical assessment of capture frequency
    
    """

    points_replace = []

    first_half_bin = []
    second_half_bin = []

    first_half_bin.append(points[start_index])


    #------------- sort definitive images -------------#
    for p, point in enumerate(points):

        if point.timestamp < points[start_index].timestamp:
            second_half_bin.append(point)
        elif point.timestamp > points[end_index].timestamp:
            first_half_bin.append(point)

        elif point.timestamp == points[start_index].timestamp or point.timestamp == points[end_index].timestamp:
            continue

        else:
            points_replace.append(point)

    if len(first_half_bin) == 0:
        # If no points are later than end slate, initialize first_half_bin
        # with points that are closest to the end slate timestamp
        first_half_bin = points_replace[-1:]
        points_replace = points_replace[:-1]

    if len(second_half_bin) == 0:
        # If no points are earlier than start slate, initialize second_half_bin
        # with points that are closest to the start slate timestamp
        second_half_bin = points_replace[:-1]
        points_replace = points_replace[-1:]



    #------------- build model for each -------------#
    try:
        diffs_first_half_mean = np.mean(np.diff(first_half_bin[1:]))
        diffs_first_half_std = np.std(np.diff(first_half_bin[1:]))

        diffs_second_half_mean = np.mean(np.diff(second_half_bin))
        diffs_second_half_std = np.std(np.diff(second_half_bin))

        diffs_combined_mean = np.mean(np.concatenate([np.diff(second_half_bin), np.diff(first_half_bin[1:])]))
        diffs_combined_std = np.std(np.concatenate([np.diff(second_half_bin), np.diff(first_half_bin[1:])]))
    except ZeroDivisionError:
        diffs_first_half_mean = 180
        diffs_first_half_std = 45
        diffs_second_half_mean = 180
        diffs_second_half_std = 45
        diffs_combined_mean = 180
        diffs_combined_std = 45
    except IndexError:
        diffs_first_half_mean = 180
        diffs_first_half_std = 45
        diffs_second_half_mean = 180
        diffs_second_half_std = 45
        diffs_combined_mean = 180
        diffs_combined_std = 45

    #------------- perform probability sort -------------#

    ## set marker ##
    fh_marker = 1
    sh_marker = -1

    ## begin by iterating through every point chrono ##
    SKIP = False
    for p, point in enumerate(points_replace):

        if SKIP: 
            SKIP = False
            continue

        ## point must either belong immediately 
        ## after last point in first_half_bin 
        ## or immediately after last point in
        ## second half bin, excluding start/end

        # probability of lying after LP in FHB 
        sigma_prob_FH = (point - first_half_bin[fh_marker-1]-diffs_first_half_mean)/diffs_first_half_std

        # probability of lying after LP in SHB
        sigma_prob_SH = (point - second_half_bin[sh_marker]-diffs_second_half_mean)/diffs_second_half_std

        print(point - first_half_bin[fh_marker-1], point - second_half_bin[sh_marker], sigma_prob_FH, sigma_prob_SH)

        # lower sigma indicates higher probability 
        if sigma_prob_FH < sigma_prob_SH:
            first_half_bin.insert(fh_marker, point)
            if p < len(points_replace)-1:
                second_half_bin.insert(sh_marker, points_replace[p+1])
        else:
            if p < len(points_replace)-1:
                second_half_bin.append(points_replace[p+1])
            first_half_bin.insert(fh_marker, point)

        SKIP = True
        fh_marker+=1

    second_half_bin.append(points[end_index])

    second_half_bin = sorted(second_half_bin, key=lambda x:x.timestamp)
    first_half_bin = sorted(first_half_bin, key=lambda x:x.timestamp)



    #------------- merge lists -------------#
    def time_plus(time, timedelta):
        start = dt.datetime(
            2000, 1, 1,
            hour=time.hour, minute=time.minute, second=time.second)
        end = start + timedelta
        end = dt.datetime(
            2000, 1, 1,
            hour=end.hour, minute=end.minute, second=end.second)
        return end

    start_time = dt.time(6,1,0)
    end_time = dt.time(7,6,0)

    bad_start_time = points[start_index].timestamp

    new_points = []
    for point in first_half_bin:
        pointcopy = copy.deepcopy(point)

        time_elapsed_bad_start = pointcopy.timestamp - bad_start_time
        pointcopy.timestamp = time_plus(start_time, time_elapsed_bad_start)
        new_points.append(pointcopy)


    bad_start_time = second_half_bin[0].timestamp
    sh_start_time = time_plus(new_points[-1].timestamp, dt.timedelta(seconds=6*diffs_combined_mean))


    for point in second_half_bin:
        pointcopy = copy.deepcopy(point)

        time_elapsed_bad_start = pointcopy.timestamp - bad_start_time
        pointcopy.timestamp = time_plus(sh_start_time, time_elapsed_bad_start)
        new_points.append(pointcopy)





    for p, point in enumerate(tqdm(new_points)):
        dst_jpg = f"{output_path}/jpgs/{p}_{point.tag}_{p}_new-time={str(point.timestamp).replace(' ', '_')}.jpg"
        dst_dng = f"{output_path}/dngs/{p}_{point.tag}_{p}_new-time={str(point.timestamp).replace(' ', '_')}.dng"

        shutil.copy2(point.fpath, dst_jpg)
        shutil.copy2(point.dng, dst_dng)

    print(f"END SLATE: {end_time}. Predicted from merge: {new_points[-1].timestamp}")


    return first_half_bin, second_half_bin, new_points, diffs_combined_mean




def suggest_reordering(points, first_half_bin, second_half_bin, output_path, start_index, end_index, diffs_combined_mean):

    """
        Get list of indices of bad images, swap their bins, and reperform sort.
    
    
        **Args**:
    
        * points (list): list of Point objects to be sorted.
        * first_half_bin (list): first half of Point objects, previously sorted.
        * second_half_bin (list): second half of Point objects, previously sorted.
        * new_points (list): merged list of sorted Point objects.
        * start_index (int): index of open slate.
        * end_index (int): index of end slate.
        * output_path (str): directory to save sorted images.
        * diffs_combined_mean: statistical assessment of capture frequency
    
        **Returns**:
    
        * new_first_half_bin (list): first half of Point objects, newly sorted.
        * new_second_half_bin (list): second half of Point objects, newly sorted.
        * new_points (list): merged list of sorted Point objects.
        * diffs_combined_mean: statistical assessment of capture frequency
    
    """
     
    

    print("Which images are bad?")
    bad_images = []
    while 1:
        num = input("Enter number, any other char will break >>> ")
        try: 
            num = int(num)
        except ValueError:
            break
        bad_images.append(num)


    fh_badims, sh_badims = [], []
    for idx in bad_images:
        if idx > len(first_half_bin):
            sh_badims.append(idx-len(first_half_bin))
        else:
            fh_badims.append(idx)


    new_second_half_bin = copy.deepcopy(second_half_bin)
    new_first_half_bin = copy.deepcopy(first_half_bin)

    for idx in sh_badims:
        new_first_half_bin.append(new_second_half_bin[idx])
    for idx in fh_badims:
        new_second_half_bin.append(new_first_half_bin[idx])


    for i in sorted(fh_badims, reverse=True):
        del new_first_half_bin[i]

    for i in sorted(sh_badims, reverse=True):
        del new_second_half_bin[i]

         


    new_second_half_bin = sorted(new_second_half_bin, key=lambda x:x.timestamp)
    new_first_half_bin = sorted(new_first_half_bin, key=lambda x:x.timestamp)


    #------------- merge lists -------------#
    def time_plus(time, timedelta):
        start = dt.datetime(
            2000, 1, 1,
            hour=time.hour, minute=time.minute, second=time.second)
        end = start + timedelta
        end = dt.datetime(
            2000, 1, 1,
            hour=end.hour, minute=end.minute, second=end.second)
        return end

    start_time = dt.time(6,1,0)
    end_time = dt.time(7,6,0)

    bad_start_time = points[start_index].timestamp

    new_points = []
    for point in new_first_half_bin:
        pointcopy = copy.deepcopy(point)

        time_elapsed_bad_start = pointcopy.timestamp - bad_start_time
        pointcopy.timestamp = time_plus(start_time, time_elapsed_bad_start)
        new_points.append(pointcopy)


    bad_start_time = new_second_half_bin[0].timestamp
    sh_start_time = time_plus(new_points[-1].timestamp, dt.timedelta(seconds=6*diffs_combined_mean))


    for point in new_second_half_bin:
        pointcopy = copy.deepcopy(point)

        time_elapsed_bad_start = pointcopy.timestamp - bad_start_time
        pointcopy.timestamp = time_plus(sh_start_time, time_elapsed_bad_start)
        new_points.append(pointcopy)




    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)
    os.mkdir(f'{output_path}/jpgs')
    os.mkdir(f'{output_path}/dngs')

    for p, point in enumerate(tqdm(new_points)):
        dst_jpg = f"{output_path}/jpgs/{p}_{point.tag}_{p}_new-time={str(point.timestamp).replace(' ', '_')}.jpg"
        dst_dng = f"{output_path}/dngs/{p}_{point.tag}_{p}_new-time={str(point.timestamp).replace(' ', '_')}.dng"

        shutil.copy2(point.fpath, dst_jpg)
        shutil.copy2(point.dng, dst_dng)


    print(f"END SLATE: {end_time}. Predicted from merge: {new_points[-1].timestamp}")


    return new_first_half_bin, new_second_half_bin, new_points, bad_images


