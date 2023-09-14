import cv2
import tensorrt as trt
import torch
import numpy as np
from collections import OrderedDict,namedtuple

import time
from visualize import plot_tracking
# from yolox.tracker.byte_tracker import BYTETracker
from tracker.byte_tracker import BYTETracker
import argparse

class TRT_engine():
    def __init__(self, weight) -> None:
        self.imgsz = [640,640]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data, int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self,im,color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img,self.r,self.dw,self.dh

    def preprocess(self,image):
        self.img,self.r,self.dw,self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img,0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict(self,img,threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores =self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        num = int(nums[0])
        new_bboxes = []
        for i in range(num):
            if(scores[i] < threshold):
                continue
            xmin = (boxes[i][0] - self.dw)/self.r
            ymin = (boxes[i][1] - self.dh)/self.r
            xmax = (boxes[i][2] - self.dw)/self.r
            ymax = (boxes[i][3] - self.dh)/self.r
            new_bboxes.append([classes[i],scores[i],xmin,ymin,xmax,ymax])
        return new_bboxes

    def predict2(self,img,threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores =self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        num = int(nums[0])
        # new_bboxes = []

        new_bboxes = []
        new_scores = []
        for i in range(num):
            if(scores[i] < threshold):
                continue
            xmin = (boxes[i][0] - self.dw)/self.r
            ymin = (boxes[i][1] - self.dh)/self.r
            xmax = (boxes[i][2] - self.dw)/self.r
            ymax = (boxes[i][3] - self.dh)/self.r
            # new_bboxes.append([classes[i],scores[i],xmin,ymin,xmax,ymax])
            new_bboxes.append([xmin,ymin,xmax,ymax])
            new_scores.append(scores[i])

        # return new_bboxes, new_scores
        return np.array(new_bboxes), np.array(new_scores)

def visualize(img,bbox_array):
    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax), (105, 237, 249), 2)
        img = cv2.putText(img, "class:"+str(clas)+" "+str(round(score,2)), (xmin,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
    return img



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument('--engine', type=str, help='Engine file')
    # parser.add_argument('--video', type=str, help='Video file')
    # parser.add_argument('--show',
    #                     action='store_true',
    #                     help='Show the detection results')
    # parser.add_argument('--out_dir',
    #                     type=str,
    #                     default='./output',
    #                     help='Path to output file')
    # parser.add_argument('--device',
    #                     type=str,
    #                     default='cuda:0',
    #                     help='TensorRT infer device')
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    args = parser.parse_args()
    return args

trt_engine = TRT_engine("../car_person_fp16.engine")

#video = cv2.VideoCapture("./test_person.mp4")
video = cv2.VideoCapture('track.mp4')

start_time = time.time()
counter = 0

frame_width = int(video.get(3))
frame_height = int(video.get(4))

out = cv2.VideoWriter('out3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))  # 保存视频


args = parse_args()
tracker = BYTETracker(args, frame_rate=30)
# timer = Timer()
frame_id = 0
results = []

t0 = time.time()
# counter = 0


if video.isOpened():
    # video.read() 一帧一帧地读取
    # open 得到的是一个布尔值，就是 True 或者 False
    # frame 得到当前这一帧的图像
    open, frame = video.read()
else:
    open = False

while open:
    ret, frame = video.read()
    # 如果读到的帧数不为空，那么就继续读取，如果为空，就退出
    if frame is None:
        break
    if ret == True:
        counter += 1  # 计算帧数

        # img = cv2.imread("./pictures/zidane.jpg")
        # results = trt_engine.predict(frame, threshold=0.5)
        # frame = visualize(frame, results)

        # cv2.putText(frame, "FPS {0}".format(counter / (time.time() - start_time)), (500, 250),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
        #             3)

        bboxes, scores = trt_engine.predict2(frame, threshold=0.5)

        # print(bboxes)
        # print(scores)
        online_targets = tracker.update(bboxes, scores)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for i, t in enumerate(online_targets):
            # tlwh = t.tlwh
            tlwh = t.tlwh_yolox
            tid = t.track_id
            # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            if tlwh[2] * tlwh[3] > args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
        t1 = time.time()
        time_ = (t1 - t0) * 1000

        online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
                                  fps=1000. / time_)

        # print("FPS: ", counter / (time.time() - t0))
        counter = 0
        t0 = time.time()

        # cv2.imshow("frame", online_im)

        out.write(online_im)
        # ch = cv2.waitKey(1)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break
    else:
        break
    frame_id += 1


        #
        # out.write(frame)  # 视频写入
        # print("FPS: ", counter / (time.time() - start_time))
        #
        #
        # counter = 0
        # start_time = time.time()

        # cv2.imshow("video", frame)
        # 这里使用 waitKey 可以控制视频的播放速度，数值越小，播放速度越快
        # 这里等于 27 也即是说按下 ESC 键即可退出该窗口
        # if cv2.waitKey(10) & 0xFF == 27:
        #     break
# video.release()
# cv2.destroyAllWindows()
