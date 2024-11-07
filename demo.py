import time
import os
import numpy as np
import uuid
import cv2
import tqdm
import shutil
import sys
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel

def main():
    # 检查命令行参数的数量
    if len(sys.argv) < 4:
        print("Usage: python demo.py <video_path> <output_video_name> <model_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_path = sys.argv[1]
    print(f"Video path is set to: {video_path}")
    audio_path = sys.argv[2]
    print(f"Audio path is set to: {audio_path}")
    output_video_name = sys.argv[3]
    print(f"output video name is set to: {output_video_name}")
    try:
        model_name = sys.argv[4]
        print(f"model_name: {model_name}")
    except Exception:
        model_name = "render.pth"

    audioModel = AudioModel()
    audioModel.loadModel("checkpoint/audio.pkl")

    renderModel = RenderModel()
    renderModel.loadModel(f"checkpoint/{model_name}")
    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)
    renderModel.reset_charactor(video_path, pkl_path)

    # wavpath = "video_data/audio0.wav"
    wavpath = audio_path
    mouth_frame = audioModel.interface_wav(wavpath)
    cap_input = cv2.VideoCapture(video_path)
    vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    cap_input.release()
    
    task_id = str(uuid.uuid1())
    os.makedirs("output/{}".format(task_id), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "output/{}/silence.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))
    for frame in tqdm.tqdm(mouth_frame):
        frame = renderModel.interface(frame)

        # 设置缩放比例
        # scale_percent = 50  # 缩放比例，50%表示缩小到原来的一半大小

        # 计算新的图像尺寸
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # 调整图像大小
        # resized_frame = cv2.resize(frame, (width, height))

        # cv2.imshow("s", resized_frame)
        # cv2.waitKey(40)

        videoWriter.write(frame)

    videoWriter.release()
    val_video = "../output/{}.mp4".format(task_id)
    os.system(
        "ffmpeg -y -i {} -i {} -c:v libx264 -loglevel quiet {}".format(save_path, wavpath, output_video_name))
    shutil.rmtree("output/{}".format(task_id))


if __name__ == "__main__":
    main()
