import os
import numpy as np
import cv2
import math


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
    M = cv2.moments(contour_in)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    # cv2.circle(img, (cx, cy), 3, (102, 204, 255), 4)  # draw centroid of contour
    return cx, cy


def midpoint(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def set_orientation(imgin, cx, cy, box):
    # line 1
    l1_p1 = box[1]
    # print l1_p1[1]
    l1_p2 = box[2]
    # line 2
    l2_p1 = box[0]
    l2_p2 = box[3]
    d_1 = np.linalg.norm(np.cross(l1_p2 - l1_p1, l1_p1 - (cx, cy)) / np.linalg.norm(
        l1_p2 - l1_p1))  # distance between l1(one side of rectangle) and centroid
    d_2 = np.linalg.norm(box[0] - box[1]) - d_1  # distance between centroid and the oppsite side of rectangle
    if d_1 > d_2:
        # cv2.line(imgin, tuple(box[3]), tuple(box[0]), (102, 204, 255), 7)
        return midpoint(box[0], box[3]), midpoint(box[1], box[2]), 1
    else:
        # cv2.line(imgin, tuple(box[1]), tuple(box[2]), (102, 204, 255), 7)
        return midpoint(box[0], box[3]), midpoint(box[1], box[2]), 0


def proximity_detect(circles, hx, hy):
    # circles: [[[x_0, y_0], radius_0]]...[[x_n, y_n], radius_n]]
    for idx, circle in [circles]:
        # print circle
        print idx
        dist = np.linalg.norm(np.array(circle[0]) - np.array([hx, hy]))
        # print dist
        # print "radius: ", circle[1]
        if dist > circle[1]:
            return False, None
        else:
            return True, idx


def track_obj(cv2_video_capture, obj_list):
    print obj_list
    fgbg = cv2.createBackgroundSubtractorKNN(history=700)
    counter = 0
    head_voter = 0
    frame_no = 0
    while (1):
        ret, frame = cv2_video_capture.read()
        if ret:
            imgin = cv2.blur(frame, (15, 15))
            fgmask = fgbg.apply(imgin)
            maskblur = cv2.blur(fgmask, (7, 7))
            ret, thresh = cv2.threshold(maskblur, 128, 255, cv2.THRESH_BINARY)
            # cv2.imshow('test1',fgmask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            # cv2.drawContours(imgin, contours, -1, (0, 255, 0), 6)
            # print fgmask.shape
            # print fgmask
            # if counter ==200:
            #     np.savetxt("foo.csv", imgin, delimiter=",")
            #     print "exported"
            # im2, contour
            # s, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(imgray, 80, 255, 0, cv2.THRESH_BINARY)
            # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            _, index = largest_contour(contours)
            try:
                # print "index is : " + str(index)
                # print len(contours[index])
                # print "=================================frame: " + str(
                #     counter) + "========================================================"
                target_contour = contours[index]
                if len(target_contour) > 10:  # Need at least 5 points to fit an ellipse
                    # centeroid coordinates
                    cx, cy = trk_cnt_centroid(imgin, target_contour)
                    ellipse = cv2.fitEllipse(target_contour)
                    # enclosing rectangle
                    box = cv2.boxPoints(ellipse)
                    box = np.int0(box)
                    (head_x, head_y), (head2_x, head2_y), vote = set_orientation(imgin, cx, cy, box)
                    myhead_x, myhead_y = 0, 0
                    head_voter = head_voter + vote
                    if counter >= 3:
                        if head_voter > 2:
                            myhead_x = head_x
                            myhead_y = head_y
                        else:
                            myhead_x = head2_x
                            myhead_y = head2_y
                        cv2.circle(imgin, (myhead_x, myhead_y), 3, (102, 204, 255), 4)
                        # res, obj_index = proximity_detect(obj_list, head_x, head_y)
                        # if res:
                        #     print "detect"
                        counter = 0
                        head_voter = 0
                        # cv2.circle(imgin, (head_x, head_y), 3, (102, 204, 255), 4)
                        # cv2.drawContours(imgin, [box], 0, (200, 0, 100), 2)
                        # cv2.ellipse(imgin, ellipse, (0, 255, 0), 2)
            except IndexError:
                print "Index Error: Possible loss of tracking"
            # cv2.drawContours(imgin, contours, index, (255, 0, 0), 6)
            cv2.putText(imgin, "Frame No. " + str(frame_no), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 250), 1, 255);
            cv2.imshow('frame', imgin)
            counter = counter + 1
            frame_no = frame_no + 1
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    cap.release()
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
            # if mode == True:
            # cv2.rectangle(imgsample,(ix,iy),(x,y),(0,255,0),-1)
            centerx = (ix - x) ** 2
            centery = (iy - y) ** 2
            radius = int(math.sqrt(centerx + centery))
            cv2.circle(imgsample, (ix, iy), radius, (0, 255, 0), -1)
            # else:
            #     cv2.circle(imgsample,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        centerx = (ix - x) ** 2
        centery = (iy - y) ** 2
        radius = int(math.sqrt(centerx + centery))
        cv2.circle(imgsample, (ix, iy), radius, (0, 255, 0), -1)
        target_obj.append([[ix, iy], radius])
        # else:
        #     cv2.circle(imgsample,(x,y),5,(0,0,255),-1)


def object_selector(video_capture):
    global reset_state, imgsample, stored_frame, target_obj
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while (1):
        if reset_state:
            imgsample = stored_frame.copy()
            target_obj = []
            reset_state = False
        else:
            cv2.imshow('image', imgsample)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("c"):
                reset_state = True
            elif k == ord(" "):
                return target_obj
            elif k == 27:
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    normpath = os.path.normpath(path)
    cap = cv2.VideoCapture('video2701.avi')
    ret, imgsample = cap.read()
    stored_frame = imgsample.copy()
    exp_obj = object_selector(cap)
    track_obj(cap, exp_obj)
    cap.release()
    cv2.destroyAllWindows()
