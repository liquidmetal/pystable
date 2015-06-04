#!/usr/bin/env python

import tornado.web
import cv2
import sys, math

import calibration_worker
import stabilize
import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize

class TrainingVideo(object):
    def __init__(self, mp4):
        self.mp4 = mp4
        self.frameInfo = []
        self.numFrames = 0
        self.duration = 0
        self.frameWidth = 0
        self.frameHeight = 0

    def readVideo(self):
        """
        Extracts the keypoints out of a video
        """
        vidcap = cv2.VideoCapture(self.mp4)

        success, frame = vidcap.read()
        prev_frame = None
        previous_timestamp = 0
        frameCount = 0

        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]

        while success:
            current_timestamp = vidcap.get(0) * 1000 * 1000
            print "Processing frame#%d (%f ns)" % (frameCount, current_timestamp)

            if prev_frame == None:
                self.frameInfo.append({'keypoints': None, 'timestamp': current_timestamp})
                prev_frame = frame
                previous_timestamp = current_timestamp
                continue

            old_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_corners = cv2.goodFeaturesToTrack(old_gray, 1000, 0.3, 30)

            if old_corners == None:
                # No keypoints?
                self.frameInfo.append({'keypoints': None, 'timestamp': current_timestamp})
                frameCount += 1
                previous_timestamp = current_timestamp
                prev_frame = frame
                success, frame = vidcap.read()
                continue

            new_corners, status, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            if len(old_corners) > 4:
                # Try and find the perfect matches
                homography, mask = cv2.findHomography(old_corners, new_corners, cv2.RANSAC, 5.0)
                mask = mask.ravel()
                new_corners_homography = np.asarray([new_corners[i] for i in xrange(len(mask)) if mask[i] == 1])
                old_corners_homography = np.asarray([old_corners[i] for i in xrange(len(mask)) if mask[i] == 1])
            else:
                new_corners_homography = new_corners
                old_corners_homography = old_corners

            if len(new_corners_homography) != len(new_corners):
                print "ELIMINATED SOME POINTS"

            # For frame n, see what keypoints existed on n-1 and
            # where they exist now
            self.frameInfo.append( {'keypoints': (old_corners_homography, new_corners_homography), 'timestamp': current_timestamp} )

            frameCount += 1
            previous_timestamp = current_timestamp
            prev_frame = frame
            success, frame = vidcap.read()

        self.numFrames = frameCount
        self.duration = current_timestamp

        return

def fetch_closest_trio(theta_x, theta_y, theta_z, timestamps, req_timestamp):
    """
    Returns the closest match for a given timestamp
    """
    try:
        if req_timestamp in timestamps:
            indexOfTimestamp = timestamps.index(req_timestamp)
            return ( (theta_x[indexOfTimestamp], theta_y[indexOfTimestamp], theta_z[indexOfTimestamp]), req_timestamp, None)
    except IndexError, e:
        import pdb; pdb.set_trace()

    i = 0
    sorted_keys = sorted(timestamps)
    for ts in sorted_keys:
        if ts > req_timestamp:
            break

        i += 1

    # We're looking for the ith and the i+1th req_timestamp
    t_previous = sorted_keys[i-1]
    t_current = sorted_keys[i]
    dt = float(t_current - t_previous)

    slope = (req_timestamp - t_previous) / dt

    t_previous_index = timestamps.index(t_previous)
    t_current_index = timestamps.index(t_current)

    new_x = theta_x[t_previous_index] * (1-slope) + theta_x[t_current_index]*slope
    new_y = theta_y[t_previous_index] * (1-slope) + theta_y[t_current_index]*slope
    new_z = theta_z[t_previous_index] * (1-slope) + theta_z[t_current_index]*slope

    return ((new_x, new_y, new_z), t_previous, t_current)

def getAccumulatedRotation(w, h, theta_x, theta_y, theta_z, timestamps, prev, current, f, gyro_delay=None, gyro_drift=None, shutter_duration=None):

    if not gyro_delay:
        gyro_delay = 0

    if not gyro_drift:
        gyro_drift = (0, 0, 0)

    if not shutter_duration:
        shutter_duration = 0

    x = np.array([[1, 0, -w/2],
                      [0, 1, -h/2],
                      [0, 0, 0],
                      [0, 0, 1]])
    A1 = np.asmatrix(x)
    transform = A1.copy()

    prev = prev + gyro_delay
    current = current + gyro_delay

    if prev in timestamps and current in timestamps:
        start_timestamp = prev
        end_timestamp = current
    else:
        (rot, start_timestamp, t_next) = fetch_closest_trio(theta_x, theta_y, theta_z, timestamps, prev)
        (rot, end_timestamp, t_next) = fetch_closest_trio(theta_x, theta_y, theta_z, timestamps, current)

    #if start_timestamp > 100000:
    #    import pdb; pdb.set_trace()

    for time in xrange(timestamps.index(start_timestamp), timestamps.index(end_timestamp)):
        time_shifted = timestamps[time] + gyro_delay
        trio, t_previous, t_current = fetch_closest_trio(theta_x, theta_y, theta_z, timestamps, time_shifted)

        gyro_drifted = (float(trio[0] + gyro_drift[0]),
                        float(trio[1] + gyro_drift[1]),
                        float(trio[2] + gyro_drift[2]))
        
        #smallR = cv2.Rodrigues(np.array([float(trio[1]), float(trio[0]), float(trio[2])]))[0]
        smallR = cv2.Rodrigues(np.array([float(gyro_drifted[1]), float(gyro_drifted[0]), float(gyro_drifted[2])]))[0]
        R = np.array([[smallR[0][0], smallR[0][1], smallR[0][2], 0],
                         [smallR[1][0], smallR[1][1], smallR[1][2], 0],
                         [smallR[2][0], smallR[2][1], smallR[2][2], 0],
                         [0,         0,         0,         1]])
        transform = R * transform
    R = getRodrigues(gyro_drifted[1], -gyro_drifted[0], -gyro_drifted[2])

    x = np.array([[1.0, 0, 0, 0],
                     [0, 1.0, 0, 0],
                     [0, 0, 1.0, f],
                     [0, 0, 0, 1.0]])
    T = np.asmatrix(x)
    x = np.array([[f, 0, w/2, 0],
                  [0, f, h/2, 0],
                  [0, 0, 1, 0]])
    transform = R*(T*transform)

    A2 = np.asmatrix(x)

    # TODO trying translation followed by rotation
    # transform = A2 * (T*transform)
    transform = A2 * transform

    return transform

def accumulateRotation(src, theta_x, theta_y, theta_z, timestamps, prev, current, f, gyro_delay=None, gyro_drift=None):
    if prev == current:
        return src

    transform = getAccumulatedRotation(src.shape[1], src.shape[0], theta_x, theta_y, theta_z, timestamps, prev, current, f, gyro_delay, gyro_drift)

    o = cv2.warpPerspective(src, transform, (src.shape[1], src.shape[0])) #, None, cv2.WARP_INVERSE_MAP)

    return o

    

def rotateImage(src, rx, ry, rz, dx, dy, dz, f, convertToRadians=False):
    if convertToRadians:
        rx = (rx) * math.pi / 180
        ry = (ry) * math.pi / 180
        rz = (rz) * math.pi / 180

    rx = float(rx)
    ry = float(ry)
    rz = float(rz)

    w = src.shape[1]
    h = src.shape[0]

    x = np.array([[1, 0, -w/2],
                      [0, 1, -h/2],
                      [0, 0, 0],
                      [0, 0, 1]])

    A1 = np.asmatrix(x)
    smallR = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
    
    R = np.array([[smallR[0][0], smallR[0][1], smallR[0][2], 0],
                     [smallR[1][0], smallR[1][1], smallR[1][2], 0],
                     [smallR[2][0], smallR[2][1], smallR[2][2], 0],
                     [0,         0,         0,         1]])


    x = np.array([[1.0, 0, 0, dx],
                     [0, 1.0, 0, dy],
                     [0, 0, 1.0, dz],
                     [0, 0, 0, 1.0]])
    T = np.asmatrix(x)

    x = np.array([[f, 0, w/2, 0],
                     [0, f, h/2, 0],
                     [0, 0, 1, 0]])
    A2 = np.asmatrix(x)

    transform = A2*(T*(R*A1))
    o = cv2.warpPerspective(src, transform, (src.shape[1], src.shape[0])) #, None, cv2.WARP_INVERSE_MAP)
    return o


def render_trio(signal_x, signal_y, signal_z, timestamps):
    plt.plot(timestamps, signal_x, 'b-', timestamps, signal_y, 'g-', timestamps, signal_z, 'r-')
    plt.ylabel("Y")
    plt.show()

def calcErrorScore(set1, set2):
    if len(set1) != len(set2):
        raise Exception("The given two sets don't have the same length")

    score = 0
    for first, second in zip(set1.tolist(), set2.tolist()):
        diff_x = math.pow(first[0][0] - second[0][0], 2)
        diff_y = math.pow(first[0][1] - second[0][1], 2)

        score += math.sqrt(diff_x + diff_y)

    return score

def calcErrorAcrossVideoObjective(parameters, videoObj, theta, timestamps):
    """
    Wrapper function for scipy
    """
    focal_length = float(parameters[0])
    gyro_delay = float(parameters[1])
    gyro_drift = ( float(parameters[2]), float(parameters[3]), float(parameters[4]) )

    return calcErrorAcrossVideo(videoObj, theta, timestamps, focal_length, gyro_delay, gyro_drift)

def calcErrorAcrossVideo(videoObj, theta, timestamps, focal_length, gyro_delay=None, gyro_drift=None, rolling_shutter=None):
    total_error = 0
    for frameCount in xrange(videoObj.numFrames):
        frameInfo = videoObj.frameInfo[frameCount]
        current_timestamp = frameInfo['timestamp']

        if frameCount == 0:
    def calcErrorAcrossVideo(self, videoObj, theta, timestamps, focal_length, gyro_delay=None, gyro_drift=None, rolling_shutter=None):
        total_error = 0
        for frameCount in xrange(videoObj.numFrames):
            frameInfo = videoObj.frameInfo[frameCount]
            current_timestamp = frameInfo['timestamp']

            if frameCount == 0:
                # INCRMENT
                #frameCount += 1
                previous_timestamp = current_timestamp
                continue

            keypoints = frameInfo['keypoints']
            if keypoints:
                old_corners = frameInfo['keypoints'][0]
                new_corners = frameInfo['keypoints'][1]
            else:
                # Don't use this for calculating errors
                continue

            # Ideally, after our transformation, we should get points from the
            # thetas to match new_corners

            #########################
            # Step 0: Work with current parameters and calculate the error score
            transform = getAccumulatedRotation(videoObj.frameWidth, videoObj.frameHeight, theta[0], theta[1], theta[2], timestamps, int(previous_timestamp), int(current_timestamp), focal_length, gyro_delay, gyro_drift, doSub=True)
            transformed_corners = cv2.perspectiveTransform(old_corners, transform)
            error = self.calcErrorScore(new_corners, transformed_corners)

            #print "Error(%d) = %f" % (frameCount, error)

            total_error += error

            # For a random frame - write out the outputs
            if frameCount == MAX_FRAMES / 2:
                img = np.zeros( (videoObj.frameHeight, videoObj.frameWidth, 3), np.uint8)
                for old, new, transformed in zip(old_corners, new_corners, transformed_corners):
                    pt_old = (old[0][0], old[0][1])
                    pt_new = (new[0][0], new[0][1])
                    pt_transformed = (transformed[0][0], transformed[0][1])
                    cv2.line(img, pt_old, pt_old, (0, 0, 255), 2)
                    cv2.line(img, pt_new, pt_new, (0, 255, 0), 1)
                    cv2.line(img, pt_transformed, pt_transformed, (0, 255, 255), 1)
                cv2.imwrite("/tmp/ddd%04d-a.png" % frameCount, img)

            # INCRMENT
            #frameCount += 1
            previous_timestamp = current_timestamp

        return total_error

        # Ideally, after our transformation, we should get points from the
        # thetas to match new_corners

        #########################
        # Step 0: Work with current parameters and calculate the error score
        transform = getAccumulatedRotation(videoObj.frameWidth, videoObj.frameHeight, theta[0], theta[1], theta[2], timestamps, int(previous_timestamp), int(current_timestamp), focal_length, gyro_delay, gyro_drift)
        transformed_corners = cv2.perspectiveTransform(old_corners, transform)
        error = calcErrorScore(new_corners, transformed_corners)

        total_error += error

        # INCRMENT
        frameCount += 1
        previous_timestamp = current_timestamp

    return total_error


def stabilize_video(mp4, csv):
    gdf = calibration_worker.GyroscopeDataFile(csv)
    gdf.parse()

    signal_x = gdf.get_signal_x()
    signal_y = gdf.get_signal_y()
    signal_z = gdf.get_signal_z()
    timestamps = gdf.get_timestamps()


    smooth_signal_x = stabilize.gaussian_filter(signal_x)
    smooth_signal_y = stabilize.gaussian_filter(signal_y)
    smooth_signal_z = stabilize.gaussian_filter(signal_z)

    # g is the difference between the smoothed version and the actual version
    g = [ [], [], [] ]
    g[0] = np.subtract(signal_x, smooth_signal_x).tolist()
    g[1] = np.subtract(signal_y, smooth_signal_y).tolist()
    g[2] = np.subtract(signal_z, smooth_signal_z).tolist()
    dgt = stabilize.diff(timestamps)

    theta = [ [], [], [] ]
    for component in [0, 1, 2]:
        sum_of_consecutives = np.add(g[component][:-1], g[component][1:])
        # The 2 is for the integration - and 10e9 for the nanosecond
        dx_0 = np.divide(sum_of_consecutives, 2 * 1000000000)
        num_0 = np.multiply(dx_0, dgt)
        theta[component] = [0]
        theta[component].extend(np.cumsum(num_0))

    #render_trio(theta[0], theta[1], theta[2], timestamps)

    # UNKNOWNS
    focal_length = 1080.0
    gyro_delay = 0
    gyro_drift = (0, 0, 0)
    shutter_duration = 0

    # Delta for calculation of unknowns
    delta_focal_length = 50.0
    delta_gyro_delay = 10 * 1000 * 1000
    delta_gyro_drift = (0.5, 0.5, 0.5)
    delta_shutter_duration = 0.5 * 1000 * 1000 * 1000

    # Termination criteria
    term_focal_length = 1.0         # 1 unit
    term_gyro_delay = 100           # 100 nanoseconds
    term_gyro_drift = 0.0001        # 0.0001 radians
    term_shutter_duration = 100     # 100 nanoseconds

    videoObj = TrainingVideo(mp4)
    videoObj.readVideo()

    print "Calibrating parameters"
    print "=====================+"

    parameters = np.asarray([1080.0, 
                                0.0,
                                0.0, 0.0, 0.0])

    result = scipy.optimize.minimize(calcErrorAcrossVideoObjective, parameters, (videoObj, theta, timestamps), 'Nelder-Mead')
    print result

    focal_length = result['x'][0]
    gyro_delay = result['x'][1]
    gyro_drift = ( result['x'][2], result['x'][3], result['x'][4] )

    print "Focal length = %f" % focal_length
    print "Gyro delay   = %f" % gyro_delay
    print "Gyro drift   = (%f, %f, %f)" % gyro_drift

    """
    overall_error = calcErrorAcrossVideo(videoObj, theta, timestamps, focal_length, gyro_delay)
    print "Error = %s" % overall_error
    while abs(delta_focal_length) > term_focal_length or abs(delta_gyro_delay) > term_gyro_delay:
        # Work on focal length
        focal_length = focal_length + delta_focal_length
        new_overall_error = calcErrorAcrossVideo(videoObj, theta, timestamps, focal_length, gyro_delay)
        print "(focal length) new Error = %s" % new_overall_error

        if new_overall_error > overall_error:
            focal_length = focal_length - delta_focal_length
            delta_focal_length = -delta_focal_length/3
        else:
            overall_error = new_overall_error

        # Work on gyro delay
        gyro_delay = gyro_delay + delta_gyro_delay
        new_overall_error = calcErrorAcrossVideo(videoObj, theta, timestamps, focal_length, gyro_delay)

        if new_overall_error > overall_error:
            gyro_delay = gyro_delay - delta_gyro_delay
            delta_gyro_delay = -delta_gyro_delay/3
        else:
            overall_error = new_overall_error
        print "(gyro delay) new Error = %s" % new_overall_error
    """
            

    # Now start reading the frames
    vidcap = cv2.VideoCapture(mp4)

    frameCount = 0
    success, frame = vidcap.read()
    previous_timestamp = 0
    while success:
        print "Processing frame %d" % frameCount
        # Timestamp in nanoseconds
        current_timestamp = vidcap.get(0) * 1000 * 1000
        print "    timestamp = %s ns" % current_timestamp
        rot, prev, current = fetch_closest_trio(theta[0], theta[1], theta[2], timestamps, current_timestamp)

        rot = accumulateRotation(frame, theta[0], theta[1], theta[2], timestamps, previous_timestamp, prev, focal_length, gyro_delay, gyro_drift)

        #print "    rotation: %f, %f, %f" % (rot[0] * 180 / math.pi, 
        #                                    rot[1] * 180 / math.pi,
        #                                    rot[2] * 180 / math.pi)
        cv2.imwrite('/tmp/frame%04d.png' % frameCount, frame)
        #rot = rotateImage(frame, theta[1][frameCount], theta[0][frameCount], theta[2][frameCount], 0, 0, 1080, 1080)
        cv2.imwrite('/tmp/rotated%04d.png' % frameCount, rot)
        frameCount += 1
        previous_timestamp = prev
        success, frame = vidcap.read()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception("Please pass the path to an mp4 file")

    mp4path = sys.argv[1]
    csvpath = mp4path.replace('.mp4', '.gyro.csv')

    stabilize_video(mp4path, csvpath)
