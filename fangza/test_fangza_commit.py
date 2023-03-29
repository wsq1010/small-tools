
import os
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


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
FANGZA_CLASSES = (
    "Wellhead",
    "GuardBoard",
    "SpoutPipe",
    "SpoutPipeGuardBoard"
)

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

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def preproc(img, input_size, swap=(2, 0, 1)):
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

g_total = 0
def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    global g_total
    # w = []
    # gb = []
    # sp = []
    # spgb = []
    class_idx = [[],[],[],[]]
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
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    gb_c = -1
    spgb_c = -1

    if len(class_idx[0]) > 0: #有井口
        gb_c = 0
        spgb_c = 0
        for j in range(len(class_idx[0])): #遍历井口
            box_w = boxes[j]
            bfind_gb = False
            bfind_spgb = True
            for k in range(len(class_idx[1])):#遍历护板
                box = boxes[k]
                cen = [(box[0]+box[2])*0.5,(box[1]+box[3])*0.5]
                if cen[0] > box_w[0] and cen[0] < box_w[2] and cen[1] > box_w[1] and cen[1] < box_w[3]:#找到护板
                    gb_c += 1
                    bfind_gb = True
                    break
            if len(class_idx[2]) > 0: #有防喷器
                # bfind = True #考虑有没有此井口防喷器的情况
                for m in range(len(class_idx[2])): #遍历防喷器
                    box = boxes[m]
                    cen = [(box[0]+box[2])*0.5,(box[1]+box[3])*0.5]
                    if cen[0] > box_w[0] and cen[0] < box_w[2] and cen[1] > box_w[1] and cen[1] < box_w[3]:#找到此井口防喷器
                        bfind_spgb = False
                        for n in range(len(class_idx[3])):#遍历防喷器护板
                            box = boxes[n]
                            cen = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                            if cen[0] > box_w[0] and cen[0] < box_w[2] and cen[1] > box_w[1] and cen[1] < box_w[3]:#找到此井口防喷器护板
                                bfind_spgb = True
                                break
                        break
                if bfind_spgb:
                    spgb_c += 1
            else:
                spgb_c += 1
            if not bfind_gb or not bfind_spgb:
                g_total += 1
                if g_total > 300:
                    cv2.rectangle(img, (box_w[0]-10, box_w[1]-10), (box_w[2]+10, box_w[3]+10), (0,0,255), 4)
                    cv2.putText(img, 'Missing Guard Board', (box_w[2]+10, box_w[3]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
            else:
                g_total = 0

    else:
        pass
    return img

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
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

def inference( img, model, num_classes, confthre, nmsthre, test_size, decoder, device='gpu' ):
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
    img, _ = preproc(img, test_size, swap=(2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    if device == "gpu":
        img = img.cuda()
    with torch.no_grad():
        # t0 = time.time()
        outputs = model(img)
        if decoder is not None:
            outputs = decoder(outputs, dtype=outputs.type())
        outputs = postprocess(
            outputs, num_classes, confthre,
            nmsthre, class_agnostic=True
        )
    # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    return outputs, img_info

def visual(output, img_info, cls_conf=0.35):
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

    vis_res = vis(img, bboxes, scores, cls, cls_conf, FANGZA_CLASSES)
    return vis_res

def image_demo(vis_folder, path, current_time, model, num_classes, confthre, nmsthre, test_size, decoder, device):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = inference(image_name, model, num_classes, confthre, nmsthre, test_size, decoder, device)
        result_image = visual(outputs[0], img_info, confthre)

        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(save_folder, os.path.basename(image_name))
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def video_demo(vis_folder, path, current_time, model, num_classes, confthre, nmsthre, test_size, decoder, device, save_result=True):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if args.save_result:
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, os.path.basename(path))
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / 4, (int(width), int(height))
    )
    count = 0
    while True:
        ret_val, frame = cap.read()
        count += 1
        if count % 4 != 0:
            continue
        if ret_val:
            outputs, img_info = inference(frame, model, num_classes, confthre, nmsthre, test_size, decoder, device)

            result_frame = visual(outputs[0], img_info, confthre)
            if save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
def push_frame(frame, vid_writer, model, num_classes, confthre, nmsthre, test_size, decoder, device, save_result=True):
    t0 = time.time()
    outputs, img_info = inference(frame, model, num_classes, confthre, nmsthre, test_size, decoder, device)
    result_frame = visual(outputs[0], img_info, confthre)
    if save_result:
        vid_writer.write(result_frame)
    logger.info("Infer time: {:.4f}s".format(time.time() - t0))
def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

def main():
    global g_total

    trt_file = 'trt_outputs/model_trt.pth' #模型文件
    test_conf = 0.6 #0.25
    nmsthre = 0.45
    test_size = (640, 640)
    device = 'gpu'
    num_classes = 4     # detect classes number of model
    depth = 0.33        # factor of model depth
    width = 0.5         # factor of model width
    act = "silu"        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
    model = YOLOX(backbone, head)
    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    if device == "gpu":
        model.cuda()
    model.eval()

    model.head.decode_in_inference = False
    decoder = model.head.decode_outputs
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_file))
    x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
    model(x)
    model = model_trt

    out_folder = './outputs'
    os.makedirs(out_folder, exist_ok=True)
    current_time = time.localtime()

    path = './Image_1784.jpg'   #测试图片
    image_demo( out_folder, path, current_time, model, num_classes, test_conf, nmsthre, test_size, decoder, device)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25
    cam = cv2.VideoCapture('rtsp://192.168.1.41:8554/ds-test')

    w,h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_folder = os.path.join(
        out_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    path = './rtsp_test.mp4'
    save_path = os.path.join(save_folder, os.path.basename(path))
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w), int(h))
    )
    count = 0

    while True:
        # img = cam.read()
        ret, img = cam.read()
        if img is None:
            print('img is None!!')
            break
        count += 1
        # if count % 4 != 0:
        #     continue
        push_frame(img, vid_writer, model, num_classes, test_conf, nmsthre, test_size, decoder, device)
        if count == 300:
            break
    vid_writer.release()
    cam.release()



if __name__ == "__main__":
    main()
    # main_test_rtsp_cv()
    # main_test_rtsp_guandao()