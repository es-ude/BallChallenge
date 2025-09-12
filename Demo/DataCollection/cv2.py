import cv2
import numpy as np
import pandas as pd
TOO_SMALL = 50

def order_points(pts):
    try:
        pts = np.reshape(pts, (4,2))
    except:
        print("No rect found")
        return (False, None)

    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return (True, rect)


def fitlerSacks(contours):
    sacks = list()
    mean = 0
    for index in range(1, len(contours)):
        cnt = contours[index]
        area = cv2.contourArea(cnt)
        mean = (mean + area) / 2

        if index == 1:
            mean = area

        if area <= mean / 2:
            break

        if area < TOO_SMALL:
            break

        sacks.append(cnt)

    return sacks

def findField(src, debug=0):
    imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    df = pd.DataFrame(hierarchy[0], columns = ["nextSameLevel", "prviousSameLevel", "FirstChild", "Parent"])
    fieldIdx = df.groupby("Parent").count().idxmax().nextSameLevel
    field = contours[fieldIdx]
    x,y,w,h = cv2.boundingRect(field)
    rect = np.array([[0,0],[w, 0],[w,h], [0, h]], np.float32)
    epsilon = 0.1*cv2.arcLength(field,True)
    approx = cv2.approxPolyDP(field, epsilon, True)
    ok, rect2 = order_points(approx)

    if not ok:
        return

    M = cv2.getPerspectiveTransform(rect2, rect)

    dst = cv2.warpPerspective(src, M, (int(w),int(h)))
    dst = cv2.resize(dst, (int(w),int(w)), cv2.INTER_AREA)

    imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 140, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    field_width = w
    sacks = fitlerSacks(contours)
    for index, sack in enumerate(sacks):
        x,y,w,h = cv2.boundingRect(sack)
        center = (x + w/2, y+h/2)
        meters = ("{:.2f}m".format(center[0]/field_width * 2),  "{:.2f}m".format(center[1]/field_width * 2))
        print("OBJ:{}, x:{}, y:{}".format(index, meters[0], meters[1]))

        if debug:
            cv2.putText(dst, str(meters), (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
            cv2.rectangle(dst, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)

    if debug:
        cv2.imshow('Objects', dst)

    if debug == 2:
        cv2.waitKey()
        return

    if debug:
        cv2.waitKey(100)


def test():
    src = cv2.imread("4.jpg")
    findField(src,debug=2)


def stream(ip):
    cap = cv2.VideoCapture('http://raspberrypi.local:8081/mjpeg')
    cv2.namedWindow('Objects', cv2.WINDOW_NORMAL)

    while(True):

        ret, frame = cap.read()
        if ret == False:
            continue
        findField(frame)

if __name__ == '__main__':
