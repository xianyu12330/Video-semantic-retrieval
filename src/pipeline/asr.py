from faster_whisper import WhisperModel
from pathlib import Path
import logging
# 添加详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_PATH = Path(__file__).parent.parent
VIDEO_PATH =  PROJECT_PATH/ "storage"
MODEL_PATH = PROJECT_PATH.parent / "models" / "faster-whisper-large-v3"
AUDIO_BUFFER = 5
logging.info("Loading Faster-Whisper model...")
model = WhisperModel(str(MODEL_PATH),device="cpu",compute_type="float16")

def asr_extract(video_id:str,window_size = 5) ->dict:
    """
        对视频音频进行 ASR 转写，并按固定时间窗口（默认5秒）进行文本切分合并。

        返回字典格式: { window_index (int) : "该窗口内的文本拼接" (str) }
        例如: { 0: "你看那只", 1: "狗狗在草地上跑得", 2: "好快" }
    """
    video_path = VIDEO_PATH / video_id
    # vad_filter=True过滤掉无声/噪音片段
    segments,info = model.transcribe(str(video_path),beam_size=5,vad_filter=False)
    logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")

    #存储文本
    window_text = {}
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        #计算跨越了哪几个5s的窗口
        start_window_idx = int(segment.start // window_size)
        end_window_idx = int(segment.end // window_size)
        #将该句话放入所有的窗口中
        for w_idx in range(start_window_idx,end_window_idx + 1):
            if w_idx not in window_text:
                window_text[w_idx] = []
            window_text[w_idx].append(text)
    #将列表字段合成为完整的句子
    for w_idx in window_text:
        window_text[w_idx] = " ".join(window_text[w_idx])
    return window_text

if __name__ == '__main__':
    test_video = VIDEO_PATH / "zhangshi.mp4"
    results = asr_extract(str(test_video))
    for w_idx, text in sorted(results.items()):
        print(f"w_idx :{w_idx} its text is {text}")
