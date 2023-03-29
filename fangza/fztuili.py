import argparse
import os.path
import time
import cv2
import numpy as np
from v5 import YOLOv5
import threading
import queue
import torch
from datetime import datetime
import random
import socket
import logging
import time
from loguru import logger
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
from yolox import YOLOX
from yolo_head import YOLOXHead
from yolo_pafpn import YOLOPAFPN
from torch2trt import TRTModule

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
FANGZA_CLASSES = (
    "Wellhead",
    "GuardBoard",
    "SpoutPipe",
    "SpoutPipeGuardBoard"
)


def logger_config(log_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def plot_one_box(x, img, color=None, label="dkerror", line_thickness=None):
    """ 画框,引自 YoLov5 工程.
    参数:
        x:      框， [x1,y1,x2,y2]
        img:    opencv图像
        color:  设置矩形框的颜色, 比如 (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class fangzaDetect(object):
    def __init__(self, model_path, fps):
        self.type = -1
        trt_file = model_path  # 'trt_outputs/model_trt.pth'  # 模型文件
        self.test_conf = 0.6  # 0.25
        self.nmsthre = 0.45
        self.test_size = (640, 640)
        self.device = 'gpu'
        self.num_classes = 4  # detect classes number of model
        depth = 0.33  # factor of model depth
        width = 0.5  # factor of model width
        act = "silu"  # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
        head = YOLOXHead(self.num_classes, width, in_channels=in_channels, act=act)
        model = YOLOX(backbone, head)
        model.apply(self.init_yolo)
        model.head.initialize_biases(1e-2)
        if self.device == "gpu":
            model.cuda()
        model.eval()

        model.head.decode_in_inference = False
        self.decoder = model.head.decode_outputs
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_file))
        x = torch.ones(1, 3, self.test_size[0], self.test_size[1]).cuda()
        model(x)
        self.model = model_trt
        # self.out_folder = out_folder
        self.g_total = 0
        self.threshold = fps * 5  # 连续违规5秒，返回录像信号

    def push_frame(self, frame):
        result_frame = self.push_frame_(frame, None, self.model, self.num_classes, self.test_conf, self.nmsthre,
                                        self.test_size, self.decoder, self.device)
        return result_frame, (self.g_total > self.threshold)

    def reset_count(self):
        self.g_total = 0

    def image_demo(self, img_path):
        current_time = time.localtime()
        self.image_demo_(None, img_path, current_time, self.model, self.num_classes, self.test_conf, self.nmsthre,
                         self.test_size, self.decoder, self.device)

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):

        # w = []
        # gb = []
        # sp = []
        # spgb = []
        class_idx = [[], [], [], []]
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            # if score < conf:
            #     continue
            #
            # class_idx[cls_id].append(i)
            if score > conf:
                class_idx[cls_id].append(i)

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        gb_c = -1
        spgb_c = -1

        if len(class_idx[0]) > 0:  # 有井口
            gb_c = 0
            spgb_c = 0
            for j in range(len(class_idx[0])):  # 遍历井口
                box_w = boxes[j]
                bfind_gb = False
                bfind_spgb = True
                for k in range(len(class_idx[1])):  # 遍历护板
                    box = boxes[k]
                    cen = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                    if cen[0] > box_w[0] and cen[0] < box_w[2] and cen[1] > box_w[1] and cen[1] < box_w[3]:  # 找到护板
                        gb_c += 1
                        bfind_gb = True
                        break
                if len(class_idx[2]) > 0:  # 有防喷器
                    # bfind = True #考虑有没有此井口防喷器的情况
                    for m in range(len(class_idx[2])):  # 遍历防喷器
                        box = boxes[m]
                        cen = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                        if cen[0] > box_w[0] and cen[0] < box_w[2] and cen[1] > box_w[1] and cen[1] < box_w[
                            3]:  # 找到此井口防喷器
                            bfind_spgb = False
                            for n in range(len(class_idx[3])):  # 遍历防喷器护板
                                box = boxes[n]
                                cen = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                                if cen[0] > box_w[0] and cen[0] < box_w[2] and cen[1] > box_w[1] and cen[1] < box_w[
                                    3]:  # 找到此井口防喷器护板
                                    bfind_spgb = True
                                    break
                            break
                    if bfind_spgb:
                        spgb_c += 1
                else:
                    spgb_c += 1
                if not bfind_gb or not bfind_spgb:
                    self.g_total += 1
                    if self.g_total > self.threshold:
                        cv2.rectangle(img, (box_w[0] - 10, box_w[1] - 10), (box_w[2] + 10, box_w[3] + 10), (0, 0, 255),
                                      4)
                        cv2.putText(img, 'Missing Guard Board', (box_w[2] + 10, box_w[3] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                else:
                    self.g_total = 0

        else:
            pass
        return img

    def push_frame_(self, frame, vid_writer, model, num_classes, confthre, nmsthre, test_size, decoder, device):
        t0 = time.time()
        outputs, img_info = self.inference(frame, model, num_classes, confthre, nmsthre, test_size, decoder, device)
        result_frame = self.visual(outputs[0], img_info, confthre)
        # if save_result:
        #     vid_writer.write(result_frame)
        logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return result_frame

    def init_yolo(self, M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self, prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

    def inference(self, img, model, num_classes, confthre, nmsthre, test_size, decoder, device='gpu'):
        # t0 = time.time()
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, test_size, swap=(2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if device == "gpu":
            img = img.cuda()
        with torch.no_grad():
            # t0 = time.time()
            outputs = model(img)
            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())
            outputs = self.postprocess(
                outputs, num_classes, confthre,
                nmsthre, class_agnostic=True
            )
        # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = self.vis(img, bboxes, scores, cls, cls_conf, FANGZA_CLASSES)
        return vis_res

    def image_demo_(self, vis_folder, path, current_time, model, num_classes, confthre, nmsthre, test_size, decoder,
                    device):
        if os.path.isdir(path):
            files = get_image_list(path)
        else:
            files = [path]
        files.sort()
        for image_name in files:
            outputs, img_info = self.inference(image_name, model, num_classes, confthre, nmsthre, test_size, decoder,
                                               device)
            result_image = self.visual(outputs[0], img_info, confthre)

            # save_folder = os.path.join(
            #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            # )
            # os.makedirs(save_folder, exist_ok=True)
            # save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format('test.jpg'))
            # cv2.imwrite(save_file_name, result_image)
            cv2.imwrite('test.jpg', result_image)
            # ch = cv2.waitKey(0)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break


def Receive(rtsp):
    cap = cv2.VideoCapture(rtsp)
    c = 0
    ret, frame = cap.read()

    while ret:
        c += 1
        ret, frame = cap.read()
        if c % 20 == 0:
            q.put(frame)


def fangza():
    print("Starting fz")
    trt_file = 'trt_outputs/model_trt1.pth'
    out_folder = "/app/rsc/data/alarms/"
    fps = 5
    fangzaD = fangzaDetect(trt_file, fps)
    count = 0
    interval_threshold = fps * 60 * 2  # 间隔2分钟
    interval_count = interval_threshold
    is_save_video = False
    save_video_num = 0  # 统计当前录像帧数
    video_length = fps * 20  # 录像时长20秒
    vid_writer = None
    global product


    if con.acquire():
        while True:
            if product is None:
                product = 'anything'


                if q.empty() != True:
                    # time.sleep(1)
                    frame = q.get()
                    count += 1
                    result_frame, is_save = fangzaD.push_frame(frame)
                    if not is_save_video:  # 当前没有录像
                        if is_save:  # True
                            if interval_count >= interval_threshold:  # 超过录像保存间隔时间
                                # save video
                                current_time = time.localtime()
                                name = time.strftime("%Y%m%d%H%M%S", current_time)
                                img_name = hostname + "_fangzacuowu_" + name
                                video_name = hostname + "_fangzacuowu_" + name
                                img_path = os.path.join(out_folder, '%s.jpg' % img_name)
                                video_path = os.path.join(out_folder, '%s.mp4' % video_name)
                                # save_path = os.path.join(save_folder, os.path.basename(path))
                                logger.info(f"video save_path is {video_path}")
                                vid_writer = cv2.VideoWriter(
                                    video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
                                )
                                cv2.imwrite(img_path, result_frame)
                                is_save_video = True
                                save_video_num = 0
                            else:
                                interval_count += 1
                    else:
                        vid_writer.write(result_frame)
                        save_video_num += 1
                        if save_video_num > video_length:
                            vid_writer.release()
                            is_save_video = False
                            interval_count = 0
                            fangzaD.reset_count()  # 充值违规帧数
                    # if count == 300:
                    #     break
            con.notify()


            con.wait()
        vid_writer.release()



def Display():
    print("Starting display")
    power = False
    global product
    print(con.acquire())
    if con.acquire():
        while True:
            if product is not None:
                product = None
                if q.empty() != True:
                    # for i in os.listdir(rtsp):
                    #         frame = cv2.imread(os.path.join(rtsp, i))
                    start = time.time()
                    frame = q.get()
                    outs = det.infer(frame)

                    print("duration: ", time.time() - start)

                    if (outs is not None) and (len(outs) > 0):
                        print("-----------------")
                        # 将结果保存为json格式
                        result["box"] = outs[0].tolist()
                        result["conf"] = outs[1].tolist()
                        result["classid"] = outs[2].tolist()

                        # 判断逻辑
                        reverse.append(frame)
                        if len(reverse) > 200:
                            reverse.clear()
                        if int(result["box"][0][3]) < 540 and int(result["classid"][0]) == 3:  #

                            now = datetime.now()
                            now_time = now.strftime("%Y%m%d%H%M%S")
                            plot_one_box(result["box"][0], frame, label="nullsafe")
                            logger.info("吊卡没有安全销")
                            cv2.imwrite(fr"/app/rsc/data/alarms/{hostname}_anquancuowu_{now_time}.jpg",
                                        frame)
                            videoWriter = cv2.VideoWriter(
                                fr"/app/rsc/data/alarms/{hostname}_anquancuowu_{now_time}.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'), 10, (1920, 1080))

                            for img in reverse:
                                print(img, "____________", type(img))
                                # print(img.shape)
                                # img = cv2.imread(i)
                                # img = cv2.resize(img,(1920, 1080))
                                videoWriter.write(img)
                            videoWriter.release()

                        if not old_box:
                            if int(result["classid"][0]) == 0 or int(result["classid"][0]) == 3:
                                old_box.append(int(result["box"][0][3]))
                                print(old_box, "_________")
                        if old_box:

                            up_down_list.append(int(result["classid"][0]))
                            # print(r"类别",int(result["classid"][0]))

                            if len(up_down_list) >= 10:
                                up_down_list.clear()

                            if int(result["box"][0][3]) - old_box[-1] >= 20 and (
                                    int(result["classid"][0]) == 0 or int(result["classid"][0]) == 3):
                                power = True
                            global exe_time
                            if datetime.now().strftime("%H") != exe_time:
                                old_box.clear()
                                power = False

                                exe_time = datetime.now().strftime("%H")

                            if power:
                                down = [i for i in up_down_list if i == 1]  #
                                # print("这是",up_down_list)
                                if len(down) > 5:  #
                                    now = datetime.now()
                                    now_time = now.strftime("%Y%m%d%H%M%S")
                                    logger.info("吊卡开口反向错误")
                                    plot_one_box(result["box"][0], frame, label="dkdown")
                                    cv2.imwrite(fr"/app/rsc/data/alarms/{hostname}_fangxiangcuowu_{now_time}.jpg", frame)
                                    videoWriter = cv2.VideoWriter(
                                        fr"/app/rsc/data/alarms/{hostname}_fangxiangcuowu_{now_time}.mp4",
                                        cv2.VideoWriter_fourcc(*'mp4v'), 5, (1920, 1080))
                                    for img in reverse:
                                        # img=cv2.imread(i)
                                        # img=cv2.resize(img,(1920,1080))
                                        videoWriter.write(img)
                                    videoWriter.release()

                con.notify()
            con.wait()


if __name__ == '__main__':
    product = None
    # 条件变量
    con = threading.Condition()
    hostname = socket.gethostname()
    old_box = []
    q = queue.Queue()
    up_down_list = []
    reverse = []
    result = {}
    det = YOLOv5()
    now_log = datetime.now()
    log_time = now_log.strftime("%Y-%m-%d")
    logger = logger_config(log_path=fr'./log/log{log_time}.txt', logging_name='小修')

    exe_time = datetime.now().strftime("%H")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", type=str,
                        default=r"rtsp:127.0.0.1:554/stream11,rtsp:127.0.0.1:554/stream12,rtsp:127.0.0.1:554/stream13")
    args = parser.parse_args()
    p1 = threading.Thread(target=Receive, args=[args.rtsp])
    p2 = threading.Thread(target=Display)
    p3 = threading.Thread(target=fangza)
    p1.start()
    p2.start()
    p3.start()
    # print(args.rtsp)

    # dir_path=r"./imagess"
    # r1,r2,r3=args.rtsp.split(",")
    # print(r1,r2,r3)
    # ta=threading.Thread(target=dfier,args=r1)
    # tb=threading.Thread(target=dfier,args=r2)
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
