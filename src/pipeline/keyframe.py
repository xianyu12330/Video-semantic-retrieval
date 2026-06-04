import time
from pathlib import Path
import ffmpeg
import cv2
import numpy as np
from sympy.plotting.textplot import is_valid

# 使用项目根目录的绝对路径（假设 src/api 的上级目录是项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
STORAGE_PATH = PROJECT_ROOT / "src" / "storage"
THRESHOLD = 6
def get_phash(frame, hash_size=32):
    frame = cv2.resize(frame, (hash_size, hash_size))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]
    average = np.mean(dct_roi)
    #将DCT系数转换为二维矩阵
    phash_1 = (dct_roi > average).astype(int)
    phash_list = phash_1.flatten().tolist()#展平
    hash_str = ''.join([str(x) for x in phash_list])
    return hash_str
#phash实现检测去重
def calculate_similarity(frame1, frame2,threshold = THRESHOLD):
    frame_hash = get_phash(frame1)
    pre_frame_hash = get_phash(frame2)
    #比较不同位数的哈希距离
    diff = sum(x1 != x2 for x1, x2 in zip(frame_hash, pre_frame_hash))
    if diff < threshold:
        return False,diff
    return True,diff


def extract_frame(video_path :str, video_id :str,debug_mode:bool=True)->list:
    frame_info = []
    output_path = STORAGE_PATH / f"keyframes/{video_id}"
    output_path.mkdir(exist_ok=True,parents=True)
    #抛弃帧的目录
    if debug_mode:
        discard_path = STORAGE_PATH/"discard"/video_id
        discard_path.mkdir(exist_ok=True,parents=True)
    # 创建 VideoCapture 对象，读取视频文件
    cap = cv2.VideoCapture(video_path)
    #获取视频文件帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    #每5s抽一帧，计算间隔
    frame_interval = int(fps * 5)
    current_frame_idx = 0
    prev_vaild_frame = None

    while cap.isOpened():
        #定位到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        #ret代表是否读取到了帧，false代表以读到结尾
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = current_frame_idx / fps
        is_valid = True
        discard_reason = ""

        #在内存中计算拉普拉斯方差
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray,cv2.CV_64F).var()
        if variance < 100:
            is_valid = False
            discard_reason = f"blur_{variance:.2f}"
        #计算与上一有效帧的相似度
        elif prev_vaild_frame is not None:
            is_similarity,diff = calculate_similarity(frame,prev_vaild_frame)
            is_valid = is_similarity
            discard_reason = f"sim_{diff}"

        #根据结果分流
        if is_valid:
            file_path = output_path / f"{timestamp:06.1f}.jpg"
            cv2.imwrite(str(file_path),frame)
            frame_info.append({
                "timestamp":timestamp,
                "framepath":str(file_path)
            })
            prev_vaild_frame = frame
        else:
            if debug_mode:
                discard_frame = discard_path / f"{timestamp:06.1f}_discarded_{discard_reason}.jpg"
                cv2.imwrite(str(discard_frame),frame)
        current_frame_idx += frame_interval #跳到下一个5s帧
    cap.release()
    return frame_info   #返回有效帧






if __name__ == '__main__':
    extract_frame("G:\\ai Infra\\project\\src\\storage\\videos\\4ded7f801937.mp4","4ded7f801937")
