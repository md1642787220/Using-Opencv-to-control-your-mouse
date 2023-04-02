import cv2
import mediapipe as mp
import time

class handDetector:  # 创建一个手部检测器类，他是根据mediapipe中的hands模块来创建的


    def __init__(self, static_image_mode = False,  # 是否将输入图像看作静态图像，False代表当作视频
        max_num_hands = 2,  # 最大检测数量
        model_complexity = 1,   #模型复杂度（0或1），landmark准确率和推理延迟通常会随着复杂度升高而升高
        min_detection_confidence = 0.5,  # 最小检测置信度
        min_tracking_confidence = 0.5 ):  # 最小追踪置信度

        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # 直接调用mediapipe的坐标绘制方法


    def findHands(self, img, draw=True):  # 其中draw=True可以决定是否想要画出landmark及其连接线
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图像色彩转换，因为hands = mpHands.Hands()这个对象只能读入RGB图像信息
        self.results = self.hands.process(imgRGB)  # 这里定义了一个实例变量
        # print(self.results.multi_hand_landmarks)  # 打印输出landmark，若未检测到，就输出None，检测到了就输出坐标值
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # 遍历每只手的坐标 共21个坐标
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                      self.mpHands.HAND_CONNECTIONS)  # 第一个参数是传入的图片，第二个参数是传入的坐标，这个方法可以将各个landmark绘制在画面中,
                # 第三个参数将各个点连在一起
        return img  # 返回已经绘制好连接点的图像
        """现在的问题在于，但我们不知道如果运用这些值。管他有没有用， 先将序号和坐标存入一个列表中，需要时就调用.因此，我们需要创建一个findPosition函数来将"""


    def findPosition(self, img, handNumber, draw=True):  # 在这里，我们并不需要图像的各项参数，只要它的尺寸，用来确定位置信息,其中handNumber表示手的序号
        lmList = []  # 定义一个空列表，用于接收后面产生的landmark序号及坐标
        # 判断是否检测到有手部信息，如果有，则输出序号及位置
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]  # 将手的序号传给myHand
            for id, lm in enumerate(myHand.landmarks):  # 将其中一只手的每个关键点的id（序号）和坐标进行列举出来
                # enumerate是python内置函数，用于枚举、列举，同时对列出的内容进行编号，因此可以同时获得索引和值，索引从0开始，
                # 其有两个参数，第一个参数为要枚举的序列，第二个参数为起始序号，默认为0开始

                # print(id, lm)

                h, w, c = img.shape  # 获取图像尺寸
                cx, cy = int(lm.x * w), int(lm.y * h)  # 坐标乘以相应尺寸,并转换成整数
                # print(id, cx, cy)
                lmList.append([id, cx, cy])  # 注意:这里lmList追加的是一个列表,追加之后就成了两级列表,如:[[id1, cx1, cy1], [id2, cx2, cy2], [id3, cx3, cy3]......]
                # """但现在的问题是，我们有了坐标值，但这些坐标值是小数，对于图片像素点来说是不好映射的，它实际上是相对于图像的比例值，因此为了能映射到
                #     实际图像上，需要乘以图像的尺寸"""
                if draw:
                    cv2.circle(img, (cx, cy), 25, (255, 255, 0), cv2.FILLED)  # 为了能明显区分每一个landmark，对其进行标注

        return lmList




def main():  # 以下代码可以用在不同项目中
    preTime = 0
    curTime = 0  # 先定义好前一帧的时间和当前时间，用以描述帧率
    cap = cv2.VideoCapture(1)
    detector = handDetector()  # 创建一个类外实例,这里无需参数，因为已经默认设置好了,注意:创建类外实例时无需self

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList =detector.findPosition(img)  # 接收序号及位置信息
        if len(lmList) != 0: # 判断列表是否为空,若不为空则打印,如果没有此判断句,当程序运行时没有检测到手,lmList就没有值,就会报错(Index out of range),因此一定要加
            print(lmList[4])  # 这里我们尝试打印4号位置(大拇指尖)的序号及坐标信息

        curTime = time.time()  # 获取当前帧的时间
        fps = 1 / (curTime - preTime)  # 帧率
        preTime = curTime  # 时间更新
        cv2.putText(img, str(int(fps)), (255, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 255), None)  # 在画面上显示帧率，并四舍五入为整数

        cv2.imshow(img)
        cv2.waitKey(1)


if __name__ == "__main__":  # 在本py文件中，这段话表明程序的入口，当单独执行本py文件的时候，if __name__ == "__main__":下面的
    # 语句会自动执行，但本py文件被当作模块导入时，if __name__ == "__main__":下面的语句不会被执行
    main()
