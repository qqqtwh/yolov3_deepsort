from kalman import *
import imutils
import time
import cv2
import numpy as np
import argparse
np.random.seed(56)


def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--counter', type=int, default=0, help='车辆总数')
    parser.add_argument('--counter_up', type=str, default=0, help='正向车道的车辆数据')
    parser.add_argument('--counter_down', type=str, default=0, help='逆向车道的车辆数据')

    parser.add_argument('--labelPath', type=str, default='./config/coco.names', help='目标类型')
    parser.add_argument('--weightsPath', type=str, default='./config/yoloV3.weights', help='权重路径')
    parser.add_argument('--configPath', type=str, default='./config/yoloV3.cfg', help='配置文件路径')


    return parser.parse_args()


# 线与线的碰撞检测：叉乘的方法判断两条线是否相交
# 计算叉乘符号
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def main(args,vs,net,ln,W,H,tracker,memory,LABELS,COLORS,line,writer):
    # 遍历每一帧图像
    while True:
        (grabed, frame) = vs.read()
        if not grabed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        # 将图像转换为blob,进行前向传播
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # 将blob送入网络
        net.setInput(blob)
        start = time.time()
        # 前向传播，进行预测，返回目标框边界和相应的概率
        layerOutputs = net.forward(ln)
        end = time.time()

        # 存放目标的检测框
        boxes = []
        # 置信度
        confidences = []
        # 目标类别
        classIDs = []

        # 遍历每个输出
        for output in layerOutputs:
            # 遍历检测结果
            for detection in output:
                # detction:1*85 [5:]表示类别，[0:4]bbox的位置信息 【5】置信度
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.3:
                    # 将检测结果与原图片进行适配
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # 左上角坐标
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)
                    # 更新目标框，置信度，类别
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # 非极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        # 检测框:左上角和右下角
        dets = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[classIDs[i]] == "car":
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    dets.append([x, y, x + w, y + h, confidences[i]])
        # 类型设置
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)

        # SORT目标跟踪
        if np.size(dets) == 0:
            continue
        else:
            tracks = tracker.update(dets)
        # 跟踪框
        boxes = []
        # 置信度
        indexIDs = []
        # 前一帧跟踪结果
        previous = memory.copy()
        memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        # 碰撞检测
        if len(boxes) > 0:
            i = int(0)
            # 遍历跟踪框
            for box in boxes:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                # 根据在上一帧和当前帧的检测结果，利用虚拟线圈完成车辆计数
                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))

                    # 利用p0,p1与line进行碰撞检测
                    if intersect(p0, p1, line[0], line[1]):
                        args.counter += 1
                        # 判断行进方向
                        if y2 > y:
                            args.counter_down += 1
                        else:
                            args.counter_up += 1
                i += 1

        # 将车辆计数的相关结果放在视频上
        cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
        cv2.putText(frame, str(args.counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
        cv2.putText(frame, str(args.counter_up), (130, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
        cv2.putText(frame, str(args.counter_down), (230, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

        # 将检测结果保存在视频
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter("./output/output.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # "释放资源"
    writer.release()
    vs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 创建跟踪器对象
    line = [(0, 150), (2560, 150)]
    tracker = Sort()
    memory = {}
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
    args = opt()

    LABELS = open(args.labelPath).read().strip().split("\n")

    net = cv2.dnn.readNetFromDarknet(args.configPath, args.weightsPath)
    # 获取yolo中每一层的名称
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # 视频
    vs = cv2.VideoCapture('./input/test.mp4')
    (W, H) = (None, None)
    writer = None
    try:
        prop = cv2.cv.CV_CAP_PROP_Frame_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("INFO:{} total Frame in video".format(total))
    except:
        print("[INFO] could not determine in video")

    main(args,vs,net,ln,W,H,tracker,memory,LABELS,COLORS,line,writer)

