import os
import numpy as np
import cv2
import math
import csv

COLOR_TRACKING = 1
MOTION_TRACKING = 0


def largest_contour(contours):
    max_area = 0.0
    contour_index = 0
    for index, contour in enumerate(contours):
        curr_area = cv2.contourArea(contour)
        if curr_area > max_area:
            max_area = curr_area
            contour_index = index
    return max_area, contour_index


def trk_cnt_centroid(img, contour_in):
    m = cv2.moments(contour_in)
    cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
    return cx, cy


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def set_orientation(imgin, cx, cy, box):
    # line 1
    l1_p1 = box[1]
    # print l1_p1[1]
    l1_p2 = box[2]
    d_1 = np.linalg.norm(np.cross(l1_p2 - l1_p1, l1_p1 - (cx, cy)) / np.linalg.norm(
        l1_p2 - l1_p1))  # distance between l1(one side of rectangle) and centroid
    d_2 = np.linalg.norm(box[0] - box[1]) - d_1  # distance between centroid and the opposite side of rectangle
    if d_1 > d_2:
        return midpoint(box[0], box[3]), midpoint(box[1], box[2]), 1
    else:
        return midpoint(box[0], box[3]), midpoint(box[1], box[2]), 0


# dot in the circles?
def proximity_detect(circles, hx, hy):
    for idx, [coordinate, radius] in enumerate(circles):
        dist = np.linalg.norm(np.array(coordinate) - np.array([hx, hy]))
        if dist < radius:
            return True, idx
    return False, None


def color_track(img_colorsep):
    hsv = cv2.cvtColor(img_colorsep, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (0, 0, 0, 0), (180, 255, 60, 0))
    # color_res = cv2.bitwise_and(img_colorsep, img_colorsep, mask=color_mask)
    maskblur = cv2.blur(color_mask, (7, 7))
    ret, thresh = cv2.threshold(maskblur, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, c_index = largest_contour(contours)
    return contours, c_index


def motion_tracking(frame):
    fgbg = cv2.createBackgroundSubtractorKNN(history=700)
    imgin = cv2.blur(frame, (15, 15))
    fgmask = fgbg.apply(imgin)
    maskblur = cv2.blur(fgmask, (7, 7))
    ret, thresh = cv2.threshold(maskblur, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, m_index = largest_contour(contours)
    return contours, m_index


frame_marked_list = []


def track_obj(working_mode, cv2_video_capture, obj_list):  # working mode: 0 for motion tracking, 1 for color tracking
    global target_obj, frame_marked_list
    # print obj_list
    frame_marked_list = []
    counter = 0
    head_voter = 0
    frame_no = 0
    # voting scheme
    totalnum = 1
    minsupport = 1
    while 1:
        ret, frame = cv2_video_capture.read()
        if ret:
            try:
                if working_mode == 0:
                    contours, index = motion_tracking(frame)
                elif working_mode == 1:
                    contours, index = color_track(frame)
                else:
                    print "unknown working mode"
                    break
                target_contour = contours[index]
                if len(target_contour) > 10:  # Need at least 5 points to fit an ellipse
                    # centeroid coordinates
                    cx, cy = trk_cnt_centroid(frame, target_contour)
                    ellipse = cv2.fitEllipse(target_contour)
                    # enclosing rectangle
                    box = cv2.boxPoints(ellipse)
                    box = np.int0(box)
                    (head_x, head_y), (head2_x, head2_y), vote = set_orientation(frame, cx, cy, box)
                    head_voter = head_voter + vote
                    if counter >= totalnum:
                        if head_voter >= minsupport:
                            myhead_x = head_x
                            myhead_y = head_y
                        else:
                            myhead_x = head2_x
                            myhead_y = head2_y
                        res, obj_index = proximity_detect(obj_list, myhead_x, myhead_y)
                        if res:
                            frame_marked_list.append(obj_index)
                        else:
                            pass  # no result returned by proximity_detect, track point is not in the selected circle
                        counter = 0
                        head_voter = 0
                    else:
                        frame_marked_list.append(0)
            except IndexError:
                pass
            counter = counter + 1
            frame_no = frame_no + 1
        else:
            break
    # print "total: ", frame_no
    # cap.release()
    target_obj = []
    cv2.destroyAllWindows()


drawing = False  # true if mouse is pressed
reset_state = False  # check if need to reset the img because 'c' is pressed by the user
ix, iy = -1, -1  # initial coordinate when Lmousebutton is pressed
target_obj = []


def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, target_obj
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            centerx = (ix - x) ** 2
            centery = (iy - y) ** 2
            radius = int(math.sqrt(centerx + centery))
            cv2.circle(img_sample, (ix, iy), radius, (0, 255, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        centerx = (ix - x) ** 2
        centery = (iy - y) ** 2
        radius = int(math.sqrt(centerx + centery))
        cv2.circle(img_sample, (ix, iy), radius, (0, 255, 0), -1)
        target_obj.append([[ix, iy], radius])


def object_selector(video_capture):
    global reset_state, img_sample, stored_frame, target_obj
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while 1:
        if reset_state:
            img_sample = stored_frame.copy()
            target_obj = []
            reset_state = False
        else:
            cv2.imshow('image', img_sample)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("c"):
                reset_state = True
            elif k == ord(" "):
                cv2.destroyWindow('image')
                return target_obj
            elif k == 27:
                break
    cv2.destroyAllWindows()


def csv_output(size, out_dict):
    print "Writing data..."
    csv_out = []
    temp = []
    total_time = 0

    for x in range(size):
        temp.append('obj' + str(x + 1))
    temp.append('total')
    csv_out.append(temp)
    temp = []
    for x in range(size):
        time = frame_marked_list.count(x) / fps
        temp.append(time)
        total_time = total_time + time
    temp.append(total_time)
    csv_out.append(temp)
    # print csv_out
    with open(os.path.join(out_dict, "out_f.csv"), 'ab') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(csv_out)


def vsplit_to_three(filename, in1, in2, cvt_state):
    print "Cutting video..."
    file_name = filename
    quoted_file_name = "\"{0}\"".format(file_name)
    total_width = 1280
    total_height = 720
    sp1 = in1
    sp2 = in2
    v1_width = sp1
    v2_width = sp2 - sp1
    v3_width = total_width - sp2
    start_v2 = sp1
    start_v3 = sp2

    path = os.path.dirname(__file__)
    normpath = os.path.normpath(path)
    output_folder = file_name + "_split"
    output_folder_path = "\"" + normpath + "\\" + output_folder
    if not cvt_state:
        v_out1 = output_folder_path + "\out1.avi\""
        v_out2 = output_folder_path + "\out2.avi\""
        v_out3 = output_folder_path + "\out3.avi\""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        command = "ffmpeg -i {0} -filter_complex \"[0:v]crop={1}:{2}:0:0[out1];\
        [0:v]crop={3}:{4}:{5}:0[out2];\
        [0:v]crop={6}:{7}:{8}:0[out3]\" \
        -map [out1] -b:v 800000 {9} \
        -map [out2] -b:v 800000 {10} \
        -map [out3] -b:v 800000 {11}".format(
            quoted_file_name, v1_width, total_height, v2_width, total_height, start_v2, v3_width, total_height,
            start_v3,
            v_out1, v_out2, v_out3)
        os.system(command)
    print "Video cutting done"
    return output_folder_path.strip("\"")


def analysis(file, out_dict):
    global img_sample, stored_frame
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(file)
    ret, img_sample = cap.read()
    stored_frame = img_sample.copy()
    print "Please select object"
    exp_obj = object_selector(cap)
    print "Analyzing..."
    track_obj(COLOR_TRACKING, cap, exp_obj)
    cap.release()
    cv2.destroyAllWindows()
    csv_output(len(exp_obj), out_dict)
    print "Done"

if __name__ == '__main__':
    f_name = 'Video 2548 - test.wmv'
    fps = 30
    c_state = 0  # 0 for conversion, 1 for no conversion
    out_path = vsplit_to_three(f_name, 320, 870, c_state)
    print "output folder: ", out_path
    for f in os.listdir(out_path):
        if f.endswith(".avi"):
            c_path = os.path.join(out_path, f)
            print "Working on: ", c_path
            print c_path
            print out_path
            analysis(c_path, out_path)