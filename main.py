import cv2
import numpy as np
import heapq
import os
from NMS import nms
from NMS import remove_little

#参数
#------------------------------------------------------------------------------------#
#单张
original_img = cv2.imread('test_images/image_1.jpg')
scale_count=2
yellow_flag=True            
is_show_process = True
#是否检测文件夹
is_detect_file = False
original_file_path = 'test_images/'
result_file_path = 'result/'
#------------------------------------------------------------------------------------#

def detect_file(original_file_path=original_file_path,result_file_path=result_file_path,scale_count=scale_count,yellow_flag=yellow_flag,is_show_process=is_show_process):
    for file in os.listdir(original_file_path):
        if file.endswith('.jpg'):
            original_img = cv2.imread(original_file_path+file)
            result_img = detect_main(original_img,scale_count,yellow_flag,is_show_process)
            #存储图片
            cv2.imwrite(result_file_path+file, result_img)
#主程序入口
def detect_main(original_img=original_img,img_count=scale_count,yellow_flag=yellow_flag,is_show_process=is_show_process):

    img_list=[]
    rect_list=[ ]
    #img_count 代表缩放次数 1280 2560 5120
    for i in range(img_count):
        img_list.append(cv2.resize(original_img, (1280*(2**i),1280*(2**i))))
    for img in img_list:
        rect_list.append(detect_cones(img,yellow_flag))
    base_img=cv2.resize(original_img, (1280,1280))#在1280*1280的图上画  
    #NMS 用于去除重叠的框 第一个参数是box 第二个参数是score 第三个参数是阈值
    #score 用面积代替
    i=0 
    all_boxes = []
    for i in range(img_count):
        for rect in rect_list[i]:
            x, y, w, h = rect
            x = int(x/(2**i))
            y = int(y/(2**i))
            w = int(w/(2**i))
            h = int(h/(2**i))
            all_boxes.append([x, y, x+w, y+h])
    #去除包含关系的框
    all_boxes = remove_little(all_boxes)
    all_boxes = np.array(all_boxes)
    #去除重叠的框
    remain_box_index = nms(all_boxes, 0.3)
    all_boxes = all_boxes[remain_box_index]
    #画框
    for box in all_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(base_img, (x1, y1), (x2, y2), (1, 255, 1), 2)
    cv2.namedWindow("final",0);
    cv2.resizeWindow("final", 640, 640);
    cv2.imshow('final', base_img)
    cv2.waitKey(0)
    return base_img

#函数定义部分
def detect_cones(img,yellow_flag=False):
    # 显示原图
    # cv2.namedWindow("ImageWindow",0);
    # cv2.resizeWindow("ImageWindow", 640, 640);
    # cv2.imshow('ImageWindow', img)
    # cv2.waitKey(0)
    # 转换至HSV色域
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义感兴趣的红色范围
    lower_red1 = np.array([0, 135, 135])
    lower_red2 = np.array([10, 255, 255])
    upper_red1 = np.array([156, 135, 150])
    upper_red2 = np.array([180, 255, 255])
    #筛选出感兴趣区域
    imgThreshLow = cv2.inRange(hsv_img, lower_red1, lower_red2)
    imgThreshHigh = cv2.inRange(hsv_img, upper_red1, upper_red2)
    threshed_img = cv2.bitwise_or(imgThreshLow, imgThreshHigh)
     #安全黄53°,99%,93% 50 80
    if yellow_flag:
        lower_yellow = np.array([30, 50, 80])
        upper_yellow = np.array([34, 255, 255])
        imgThreshYellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        threshed_img = cv2.bitwise_or(threshed_img, imgThreshYellow)

    # 定义开闭运算kernel,这里使用了非规则的kernel来增强性能

    kernel_dilate = np.ones((15,3),np.uint8)
    kernel_erode_h= np.ones((5,1),np.uint8)
    kernel_erode_w = np.ones((1,5),np.uint8)

    # 进行开闭运算
    # 先进行垂直方向的开运算，将相连的路障分开
    smoothed_img = cv2.erode(threshed_img, kernel_erode_h, iterations = 7)
    smoothed_img = cv2.dilate(smoothed_img, kernel_erode_h, iterations = 5)


    # 再进行水平方向的开运算，进一步去掉噪声
    smoothed_img = cv2.erode(smoothed_img, kernel_erode_w, iterations = 2)
    # smoothed_img = cv2.dilate(smoothed_img, kernel_erode_w, iterations = 2)

    # 最后进行膨胀操作，使得障碍物区域更加连续 闭运算
    smoothed_img = cv2.dilate(smoothed_img, kernel_dilate, iterations = 9)
    mask = cv2.erode(smoothed_img, kernel_erode_w, iterations = 4)

    # 显示处理后的图像
    if is_show_process:

        cv2.namedWindow("ori",0);
        cv2.resizeWindow("ori", 640, 640);
        cv2.imshow('ori', threshed_img)
        cv2.waitKey(0)

        cv2.namedWindow("proceed",0);
        cv2.resizeWindow("proceed", 640, 640);
        cv2.imshow('proceed', mask)
        cv2.waitKey(0)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = np.zeros_like(img)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)


    approxContours = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        approxContours.append(approx)

    img_Contours = np.zeros_like(mask) 
    cv2.drawContours(img_Contours, approxContours, -1, 255, 1)


    allConvexHulls = []

    for approxContour in approxContours:
        allConvexHulls.append(cv2.convexHull(approxContour))

    imgAllConvexHulls = np.zeros_like(mask)
    cv2.drawContours(imgAllConvexHulls, allConvexHulls, -1, (255, 255, 255), 2)


    convexHull3To15 = []

    for convexHull in allConvexHulls:
        if 3 <= len(convexHull) <= 15:
            convexHull3To15.append(cv2.convexHull(convexHull))

    imgConvexHulls3To10 = np.zeros_like(mask)
    cv2.drawContours(imgConvexHulls3To10, convexHull3To15, -1, (255, 255, 255), 2)

    cones = []
    bounding_Rects = []

    for ch in convexHull3To15:
        if convexHullPointingUp(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_Rects.append(rect)
    imgTrafficCones = np.zeros_like(mask)
    # cv2.drawContours(imgTrafficCones, cones, -1, (255, 255, 255), 2)

    # cv2.namedWindow("mask",0);
    # cv2.resizeWindow("mask", 640, 640);
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # # cv2.imshow('canny', canny)
    # # cv2.imshow('res', res)
    # # cv2.waitKey(0)
    cv2.namedWindow("contours",0);
    cv2.resizeWindow("contours", 640, 640);
    cv2.imshow('contours', imgContours)
    cv2.waitKey(0)
    cv2.namedWindow("approxContours",0);
    cv2.resizeWindow("approxContours", 640, 640);
    cv2.imshow('approxContours', img_Contours)
    cv2.waitKey(0)
    cv2.namedWindow("convexHull",0);
    cv2.resizeWindow("convexHull", 640, 640);
    cv2.imshow('convexHull', imgAllConvexHulls)
    cv2.waitKey(0)
    cv2.namedWindow("convexHull3To15",0);
    cv2.resizeWindow("convexHull3To15", 640, 640);
    cv2.imshow('convexHull3To15', imgConvexHulls3To10)
    cv2.waitKey(0)
    # cv2.namedWindow("Up Cones",0);
    # cv2.resizeWindow("Up Cones", 640, 640);
    # cv2.imshow('Up Cones', imgTrafficCones)
    # cv2.waitKey(0)
    # cv2.namedWindow("final",0);
    # cv2.resizeWindow("final", 640, 640);
    # cv2.imshow('final', finalcopy)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bounding_Rects

    # cv2.imshow('Cone', img)
# cv2.waitKey(0)

    # imgTrafficCones = np.zeros_like(mask)
    # cv2.drawContours(imgTrafficCones, cones, -1, (255, 255, 255), 2)

    # finalcopy = img.copy()

    # for rect in bounding_Rects:
    #     cv2.rectangle(finalcopy, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 2) 

def convexHullPointingUp(ch):
    pointsAboveCenter, poinstBelowCenter = [], []

    x, y, w, h = cv2.boundingRect(ch)
    aspectRatio = w / h

    if aspectRatio < 0.8:
        verticalCenter = y + h / 2

        for point in ch:
            if point[0][1] < verticalCenter:
                pointsAboveCenter.append(point)
            elif point[0][1] >= verticalCenter:
                poinstBelowCenter.append(point)
        if pointsAboveCenter.__len__() < 2:
            return True
        leftX = poinstBelowCenter[0][0][0]
        rightX = poinstBelowCenter[0][0][0]
        for point in poinstBelowCenter:
            if point[0][0] < leftX:
                leftX = point[0][0]
            if point[0][0] > rightX:
                rightX = point[0][0]

        # for point in pointsAboveCenter:
        #     if (point[0][0] < leftX) or (point[0][0] > rightX):
        #         return False
        #根据Y找到最高的两个点 判断是否在leftx到rightx之间
        pointY = []
        for point in pointsAboveCenter:
            pointY.append(point[0][1])
        top_number = heapq.nsmallest(2, pointY) 
        top_index = []
        for t in top_number:
            index = pointY.index(t)
            top_index.append(index)
            pointY[index] = 0
        topPoint = pointsAboveCenter[top_index[0]]
        secondPoint = pointsAboveCenter[top_index[1]]
        if (topPoint[0][0] < leftX) or (topPoint[0][0] > rightX) or (secondPoint[0][0] < leftX) or (secondPoint[0][0] > rightX):
            return False
        # if abs(topPoint[0][0] - secondPoint[0][0]) > abs(leftX-rightX):
        #     return False
        #如果最高两个点的距离小于底部距离的一半则认为是锥体，如果最高两个点都在底部内则认为是锥体，否则不是锥体
        # if abs(topPoint[0][0] - secondPoint[0][0]) > abs(leftX-rightX)*0.9:
        #     return False
        # elif (topPoint[0][0] > leftX) and (topPoint[0][0] < rightX) and (secondPoint[0][0] > leftX) and (secondPoint[0][0] < rightX):
        #     return True
        # else:
        #     return False

    else:
        return False

    return True


# # cv2.imshow('Cone', img)
# # cv2.waitKey(0)
# cv2.namedWindow("mask",0);
# cv2.resizeWindow("mask", 640, 640);
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# # cv2.imshow('canny', canny)
# # cv2.imshow('res', res)
# # cv2.waitKey(0)
# cv2.namedWindow("contours",0);
# cv2.resizeWindow("contours", 640, 640);
# cv2.imshow('contours', imgContours)
# cv2.waitKey(0)
# cv2.namedWindow("approxContours",0);
# cv2.resizeWindow("approxContours", 640, 640);
# cv2.imshow('approxContours', img_Contours)
# cv2.waitKey(0)
# cv2.namedWindow("convexHull",0);
# cv2.resizeWindow("convexHull", 640, 640);
# cv2.imshow('convexHull', imgAllConvexHulls)
# cv2.waitKey(0)
# cv2.namedWindow("convexHull3To15",0);
# cv2.resizeWindow("convexHull3To15", 640, 640);
# cv2.imshow('convexHull3To15', imgConvexHulls3To10)
# cv2.waitKey(0)
# cv2.namedWindow("Up Cones",0);
# cv2.resizeWindow("Up Cones", 640, 640);
# cv2.imshow('Up Cones', imgTrafficCones)
# cv2.waitKey(0)
# cv2.namedWindow("final",0);
# cv2.resizeWindow("final", 640, 640);
# cv2.imshow('final', finalcopy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
if __name__ == '__main__':
    if is_detect_file:
        detect_file()
    else:
        detect_main()