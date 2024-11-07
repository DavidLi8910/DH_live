import argparse
import os
import shutil
import logging
import subprocess
import sys
import glob
import math
import pickle

import torch
import tqdm
from basicsr import imwrite

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
# sys.path.append("..")

import mediapipe as mp

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from talkingface.util.smooth import smooth_array
from talkingface.run_utils import calc_face_mat
from talkingface.utils import *
from gfpgan import GFPGANer

# 配置logging
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_path = os.path.join(log_dir, 'data_preparation_face.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_path)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 warning 和 error 信息
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # 只显示 error 信息

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8

def load_model(version='1.3', upscale=1, bg_upsampler_model='realesrgan', bg_tile=400):
    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler_model == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    return restorer

def run_gfpgan(model, input, output_dir, only_center_face=False, aligned=False, weight=0.5):

    # ------------------------ input & output ------------------------
    if input.endswith('/'):
        input = input[:-1]
    if os.path.isfile(input):
        img_list = [input]
    else:
        img_list = sorted(glob.glob(os.path.join(input, '*')))

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        # print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        _, _, restored_img = model.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight)

        # save restored img
        if restored_img is not None:
            extension = ext[1:]
            save_restore_path = os.path.join(output_dir, f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{output_dir}] folder.')

def detect_face(frame):
    # 剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80的
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections or len(results.detections) > 1:
            logger.warning("No face detected or multiple faces detected")
            return -1, None

        rect = results.detections[0].location_data.relative_bounding_box
        out_rect = [rect.xmin, rect.xmin + rect.width, rect.ymin, rect.ymin + rect.height]

        nose_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.NOSE_TIP)
        l_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.LEFT_EYE)
        r_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        # print(nose_, l_eye_, r_eye_)

        if nose_ is None or l_eye_ is None or r_eye_ is None:
            return -2, out_rect

        if nose_.x > l_eye_.x or nose_.x < r_eye_.x:
            return -3, out_rect

        h, w = frame.shape[:2]
        # print(frame.shape)
        if rect.xmin < 0 or rect.ymin < 0 or rect.xmin + rect.width > w or rect.ymin + rect.height > h:
            return -4, out_rect

        if rect.width * w < 100 or rect.height * h < 100:
            return -5, out_rect

    return 1, out_rect


def calc_face_interact(face0, face1):
    """
    计算两个面部区域之间的交互程度。

    参数:
    face0 (tuple): 第一个面部区域的坐标，格式为 (x_min, x_max, y_min, y_max)。
    face1 (tuple): 第二个面部区域的坐标，格式为 (x_min, x_max, y_min, y_max)。

    返回:
    float: 两个面部区域在交集区域内的面积占比最小的那个值，表示交互程度。
    """
    # 计算两个面部区域的交集边界
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])

    # 检查交集区域是否有效
    if x_max <= x_min or y_max <= y_min:
        return 0  # 如果没有交集，交互程度为0

    # 计算交集区域的宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    # 计算两个面部区域在交集区域内的面积占比
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / (width * height)
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / (width * height)

    # 返回两个面部区域中面积占比最小的那个
    return min(tmp0, tmp1)


def detect_face_mesh(frame, min_detection_confidence=0.6):
    """
    使用MediaPipe FaceMesh模型检测视频帧中的人脸，并提取出人脸的3D关键点。

    参数:
    frame (numpy.ndarray): 输入的视频帧。
    min_detection_conf议confidence (float): 人脸检测的最小置信度阈值。

    返回:
    numpy.ndarray: 包含人脸3D关键点的数组，如果没有检测到人脸则返回全零数组。
    """
    pts_3d = np.zeros([478, 3])
    try:
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.error(f"FaceMesh load fail: {e}")
        return pts_3d

    if not results.multi_face_landmarks:
        logger.warning("No face detected")
    else:
        image_height, image_width = frame.shape[:2]
        for face_landmarks in results.multi_face_landmarks:
            for index_, landmark in enumerate(face_landmarks.landmark):
                x_pixel = min(math.floor(landmark.x * image_width), image_width - 1)
                y_pixel = min(math.floor(landmark.y * image_height), image_height - 1)
                z_pixel = min(math.floor(landmark.z * max(image_width, image_height)),
                               max(image_width, image_height) - 1)
                pts_3d[index_] = np.array([x_pixel, y_pixel, z_pixel])

    return pts_3d


def is_frame_completely_black(frame):
    # 将帧转换为灰度
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算灰度帧中的非零像素数量
    non_zero_pixels = cv2.countNonZero(gray_frame)
    # 如果没有非零像素，则帧完全为黑色
    return non_zero_pixels == 0

def ExtractFromVideo(video_path, circle=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error: Cannot open video file {}".format(video_path))
        return -1

    dir_path = os.path.dirname(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    totalFrames = int(totalFrames)
    pts_3d = np.zeros([totalFrames, 478, 3])
    face_rect_list = []

    # os.makedirs("../preparation/{}/image".format(model_name))
    # for frame_index in tqdm.tqdm(range(totalFrames)):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=total_frames, desc="Processing frames")

    frame_index = 0
    while True:
        ret, frame = cap.read()  # 按帧读取视频
        # #到视频结尾时终止
        if not ret:
            break

        # cv2.imwrite("../preparation/{}/image/{:0>6d}.png".format(model_name, frame_index), frame)
        tag_, rect = detect_face(frame)
        if frame_index == 0 and tag_ != 1:
            logger.error("First frame face detection abnormal."
                         "Please exclude multiple faces, "
                         "large angle side faces (nose not between two eyes), "
                         "some face frames outside the screen, and face pixels below 80 * 80")

            pts_3d = -2
            cap.release()
            break
        elif tag_ == -1:  # 有时候人脸检测会失败，就用上一帧的结果替代这一帧的结果
            '''
            # 检测帧是否完全为黑色，如果是黑色则舍弃此视频
            if is_frame_completely_black(frame):
                print("第{}帧人脸检测异常，为全黑图像无人脸, tag: {}".format(frame_index, tag_))
                pts_3d = -3
                cap.release()
                break

            rect = face_rect_list[-1]
            '''
            logger.warning("Face detection failed.")
            pts_3d = -4
            cap.release()
            break
        elif tag_ != 1:
            logger.error("Face detection abnormal")
            pts_3d = -5
            cap.release()
            break

        if len(face_rect_list) > 0:
            face_area_inter = calc_face_interact(face_rect_list[-1], rect)
            # print(frame_index, face_area_inter)
            if face_area_inter < 0.6:
                logger.error("The amplitude of the facial area change is too large, "
                             "please double check. The value exceeds {}, "
                             "frame_num: {}".format(face_area_inter, frame_index))
                pts_3d = -6
                cap.release()
                break

        face_rect_list.append(rect)

        x_min = rect[0] * vid_width
        y_min = rect[2] * vid_height
        x_max = rect[1] * vid_width
        y_max = rect[3] * vid_height
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        # x_min = int(max(0, x_mid - seq_w * 0.65))
        # y_min = int(max(0, y_mid - seq_h * 0.4))
        # x_max = int(min(vid_width, x_mid + seq_w * 0.65))
        # y_max = int(min(vid_height, y_mid + seq_h * 0.8))
        crop_size = int(max(seq_w * 1.35, seq_h * 1.35))
        x_min = int(max(0, x_mid - crop_size * 0.5))
        y_min = int(max(0, y_mid - crop_size * 0.45))
        x_max = int(min(vid_width, x_min + crop_size))
        y_max = int(min(vid_height, y_min + crop_size))

        frame_face = frame[y_min:y_max, x_min:x_max]
        # cv2.imshow("s", frame_face)
        # cv2.waitKey(20)
        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])

        pbar.update(1)
        frame_index += 1

    cap.release()  # 释放视频对象
    return pts_3d


def process_video(video_path, output_dir, export_imgs=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error: Cannot open video file {}".format(video_path))
        return 0
    try:
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
        totalFrames = int(totalFrames)
        cap.release()
        pts_3d = ExtractFromVideo(video_path)

        if type(pts_3d) is not np.ndarray or len(pts_3d) != totalFrames:
            logger.error("Error in video: {}".format(video_path))
            if os.path.exists(video_path):
                os.remove(video_path)
            return

        video_name = os.path.basename(video_path).split(".")[0]
        # video_data_path = os.path.join(os.path.dirname(video_path) + "_crop_face", video_name)
        os.makedirs(output_dir, exist_ok=True)

        if export_imgs:
            # 计算整个视频中人脸的范围
            x_min, y_min, x_max, y_max = np.min(pts_3d[:, :, 0]), np.min(
                pts_3d[:, :, 1]), np.max(
                pts_3d[:, :, 0]), np.max(pts_3d[:, :, 1])
            new_w = int((x_max - x_min) * 0.55) * 2
            new_h = int((y_max - y_min) * 0.6) * 2
            center_x = int((x_max + x_min) / 2.)
            center_y = int(y_min + (y_max - y_min) * 0.6)
            size = max(new_h, new_w)
            x_min, y_min, x_max, y_max = int(center_x - size // 2), int(center_y - size // 2), int(
                center_x + size // 2), int(center_y + size // 2)

            # 确定裁剪区域上边top和左边left坐标
            top = y_min
            left = x_min
            # 裁剪区域与原图的重合区域
            top_coincidence = int(max(top, 0))
            bottom_coincidence = int(min(y_max, vid_height))
            left_coincidence = int(max(left, 0))
            right_coincidence = int(min(x_max, vid_width))
            logger.info("Face range：{}:{}, {}:{}".format(top_coincidence, bottom_coincidence, left_coincidence, right_coincidence))

            out_size = 512
            scale = 512. / size
            pts_3d = (pts_3d - np.array([left_coincidence, top_coincidence, 0])) * scale
            path_output_pkl = "{}/keypoint_rotate.pkl".format(output_dir)
            with open(path_output_pkl, "wb") as f:
                pickle.dump(pts_3d, f)
            os.makedirs("{}/image".format(output_dir), exist_ok=True)
            ffmpeg_cmd = "ffmpeg -i {} -vf crop={}:{}:{}:{},scale=512:512:flags=neighbor -loglevel quiet -y {}/image/%06d.png".format(
                video_path,
                right_coincidence - left_coincidence,
                bottom_coincidence - top_coincidence,
                left_coincidence,
                top_coincidence,
                output_dir
            )
            subprocess.run(ffmpeg_cmd, shell=True, check=True)

            gfpgan_model = load_model(version='1.3', upscale=1, bg_upsampler_model='realesrgan', bg_tile=400)
            run_gfpgan(model=gfpgan_model, input="{}/image".format(output_dir),
                       output_dir="{}/image".format(output_dir), weight=0.5)

        # 读取图像文件列表并排序
        img_filelist = glob.glob("{}/image/*.png".format(output_dir))
        img_filelist.sort()

        # 加载pickle文件中的图像信息
        path_output_pkl = "{}/keypoint_rotate.pkl".format(output_dir)
        with open(path_output_pkl, "rb") as f:
            images_info = pickle.load(f)[:, main_keypoints_index, :]

        # 平滑数组并重塑形状
        pts_driven = images_info.reshape(len(images_info), -1)
        pts_driven = smooth_array(pts_driven).reshape(len(pts_driven), -1, 3)

        # 加载面部位姿跟踪矩阵的平均值
        face_pts_mean = np.loadtxt(os.path.join(current_dir, "../data/face_pts_mean_mainKps.txt"))

        # 计算面部位姿跟踪矩阵
        try:
            mat_list, pts_normalized_list, face_pts_mean_personal = calc_face_mat(pts_driven, face_pts_mean)
        except Exception as e:
            logger.error("error in video: {}".format(video_path))
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                shutil.rmtree(output_dir)

            if os.path.exists(video_path):
                os.remove(video_path)
            return

        pts_normalized_list = np.array(pts_normalized_list)
        # print(face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1])
        # print(np.max(pts_normalized_list[:,INDEX_FACE_OVAL[:10], 1], axis = 1))
        face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 1],
                                                                 axis=0) + np.arange(5, 25, 2)
        face_pts_mean_personal[INDEX_FACE_OVAL[:10], 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0],
                                                                 axis=0) - (9 - np.arange(0, 10))
        face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1],
                                                                  axis=0) - np.arange(5, 25, 2) + 28
        face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0],
                                                                  axis=0) + np.arange(0, 10)

        face_pts_mean_personal[INDEX_FACE_OVAL[10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[10], 1], axis=0) + 25

        # for keypoints_normalized in pts_normalized_list:
        #     img = np.zeros([1000,1000,3], dtype=np.uint8)
        #     for coor in face_pts_mean_personal:
        #         # coor = (coor +1 )/2.
        #         cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, (255, 0, 0), thickness)
        #     for coor in keypoints_normalized:
        #         # coor = (coor +1 )/2.
        #         cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
        #     cv2.imshow("a", img)
        #     cv2.waitKey(30)

        with open("{}/face_mat_mask.pkl".format(output_dir), "wb") as f:
            pickle.dump([mat_list, face_pts_mean_personal], f)
    except Exception as e:
        logger.error("Error processing {}: {}".format(video_path, e))
    finally:
        cap.release()


def main(args):
    if not os.path.isdir(args.videos_dir):
        raise NotADirectoryError("The specified path is not a directory")

    # 获取video_name参数
    logger.info("Video dir is set to: {}".format(args.videos_dir))

    # data_dir = r"F:\C\AI\CV\88"
    video_files = glob.glob("{}/*.mp4".format(args.videos_dir))
    if not video_files:
        raise FileNotFoundError("Empty directory")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for video_path in video_files:
            video_name = os.path.basename(video_path).split(".")[0]
            video_data_path = os.path.join(os.path.dirname(video_path) + "_crop_face", video_name)

            try:
                executor.submit(process_video, video_path, video_data_path)
            except Exception as e:
                logger.error("Error processing {}: {}".format(video_path, e))
                if os.path.exists(video_data_path):
                    shutil.rmtree(video_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Script")
    parser.add_argument("--videos_dir", type=str, help="Directory containing video files")
    args = parser.parse_args()
    main(args)
