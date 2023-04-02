import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # 直接调用mediapipe的坐标绘制方法

preTime = 0
curTime = 0  # 先定义好前一帧的时间和当前时间，用以描述帧率

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图像色彩转换，因为hands = mpHands.Hands()这个对象只能读入RGB图像信息
    results = hands.process(imgRGB)
    # print(reults.multi_hand_landmarks)  # 打印输出landmark，若未检测到，就输出None，检测到了就输出坐标值
    if results.mult_hand_landmarks:
        for handLms in results.mult_hand_landmarks:  # 遍历每只手的坐标 共21个坐标
            for id, lm in enumerate(handLms.landmarks):  # 将其中一只手的每个关键点的id（序号）和坐标进行列举出来
                # enumerate是python内置函数，用于枚举、列举，同时对列出的内容进行编号，因此可以同时获得索引和值，索引从0开始，
                # 其有两个参数，第一个参数为要枚举的序列，第二个参数为起始序号，默认为0开始

                # print(id, lm)

                h, w, c = img.shape  # 获取图像尺寸
                cx, cy = int(lm.x * w), int(lm.y * h)  # 坐标乘以相应尺寸,并转换成整数
                print(id, cx, cy)
                # """但现在的问题是，我们有了坐标值，但这些坐标值是小数，对于图片像素点来说是不好映射的，它实际上是相对于图像的比例值，因此为了能映射到
                #     实际图像上，需要乘以图像的尺寸"""

                cv2.circle(img, (cx, cy), 25, (255, 255, 0))  # 为了能明显区分每一个landmark，对其进行标注

            mpDraw.draw_landmarks(img, handLms,
                                  mpHands.HAND_CONNECTIONS)  # 第一个参数是传入的图片，第二个参数是传入的坐标，这个方法可以将各个landmark绘制在画面中,
            # 第三个参数将各个点连在一起
            """现在的问题在于，但我们不知道如果运用这些值。管他有没有用， 先将序号和坐标存入一个列表中，需要时就调用"""

    curTime = time.time()  # 获取当前帧的时间
    fps = 1 / (curTime - preTime)  # 帧率
    preTime = curTime  # 时间更新
    cv2.putText(img, str(int(fps)), (255, 10), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), None)  # 在画面上显示帧率，并四舍五入为整数

    cv2.imshow(img)
    cv2.waitKey(1)
