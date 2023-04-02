import numpy as np

import handTrackModule as htm
import cv2
import numpy
import time
import autopy  # 它包括用于控制键盘和鼠标，在屏幕上查找颜色和位图以及显示警报的功能

###################################
wCam, hCam = 640, 480  # 定义画面尺寸
framR = 100  # 限定的尺寸
smoothening = 7  # 平滑度
###################################
cap = cv2.VideoCapture(1)

cap.set(3, wCam)
cap.set(4, hCam)  # 定义画面尺寸
pTime = 0

preLocaX, preLocaY = 0, 0  # 先前的坐标值
curLocaX, curLocaY = 0, 0  # 当前的坐标值

detector = htm.handDetector(maxHands=1)  # 检测器，检测最大数量
wScr, hScr = autopy.screen.size()  # 自动获取屏幕尺寸

while True:
    # 1. find hand landmarks
    success, img = cap.read()  # 读取视频内容
    img = detector.findHands(img)  # 如果想要画出landmark及其连线,加入draw=True
    lmlist, bbox = detector.findPosition(img)  # 获取整只手的坐标，存入lmlist，bbox为边界框数组；findPosition(img)中有两个元组，一个代表lmlist，一个代表bbox
    # 如果想要画出●圆圈作为标记点,在后面加draw=True

    # 2. 获得食指和中指的指尖
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # 食指指尖坐标
        x2, y2 = lmlist[12][1:]  # 中指指尖坐标

        print(x1, y1, x2, y2)

    # 3. 检查哪个指尖向上

    fingers = detector.fingersUp()  # 将5个值作为数组传给fingers[1，1，1，1，1] 其中1代表指尖向上，0表示无指尖，依次代表大拇指到小拇指
    cv2.rectangle(img, (framR, framR), (wCam - framR, hCam - framR), (255, 255, 0), 2)  # 画框，设置框尺寸 rectangle参数：图像源，左上顶点坐标，右下对角线顶点坐标，颜色，粗细，线条类型


    # 4. 仅检测到食指：移动模式

    if fingers[1] == 1 and fingers[2] == 0:  # 食指为1，中指为0

        # 5. 转换坐标，用来检测食指移动到哪里，然后将坐标发送给鼠标

        x3 = np.interp(x1, (framR, wCam - framR), (0, wScr))  # 线性插值，将手指相对于摄像头画面的位置转换为鼠标相对于屏幕的位置
        y3 = np.interp(y1, (framR, hCam - framR), (0, hScr))

    # 6. 设置平滑值

    curLocaX = preLocaX + (x3 - preLocaX) / smoothening  # 平滑处理，如果smoothening越大，动作越平滑，但数值大会产生运动过慢和运动滞后的问题，
    curLocaY = preLocaY + (y3 - preLocaY) / smoothening
    # 7. 移动鼠标

    autopy.mouse.move(wScr - curLocaX, curLocaY)  # 将转换过的坐标发送给鼠标
    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # 以当前坐标为圆心画圆圈，用以表示手指位置
    preLocaX, preLocaY = curLocaX, curLocaY  # 对位置坐标进行迭代
    """但出现了一个问题，当手往下移动的时候，由于检测不到手的全貌，导致鼠标无法停留在屏幕最下方，因此需要划定一个操作区域，于是设置 framR=100 """

    # 8. 食指和中指都向上，点击模式
    if fingers[1] == 1 and fingers[2] == 1:  # 食指为1，中指也为1
        # 9. 检测指间距离
        length, img, infoLine, = detector.findDistance(8, 12, img)  # infoLine表示信息线  findDistance(点1, 点2, img)  计算点1和点2 之间的距离
        # 10. 如果距离短，点击鼠标
        if length < 40:  # 设定一个阈值，当两指指尖小于这个阈值，才被视为点击模式
            cv2.circle(img, (infoLine[4], infoLine[5]), 15, (0, 255, 255), cv2.FILLED)   # 在两指之间画圈，其中(infoLine[4], infoLine[5])表示两指之间圆心坐标
            autopy.mouse.click()  # 点击

    """但问题在于，由于是一帧一帧进行检测，坐标会发生抖动，最终会使点击不准确，于是设置平滑度，第6步"""




    # 11. 帧率

    cTime = time.time()  # 获取当前时间
    fps = 1 / (cTime - pTime)  # 获取帧率
    pTime = cTime
    cv2.putText(img, str(int(fps)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)  # 设置文字及其格式
    # 12. 显示
    cv2.namedWindow()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
