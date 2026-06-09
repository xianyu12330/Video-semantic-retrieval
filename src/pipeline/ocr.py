
from paddleocr import PaddleOCR
import os
from pathlib import Path
import logging
PROJECT_PATH = Path(__file__).parent.parent
logging.info("Loading PaddleOCR model for language detection...")
ocr_model = PaddleOCR(
    use_doc_unwarping=False,
    lang="ch",
)
logging.info("PaddleOCR model loaded.")

def process_ocr(video_id:str,window_size:int = 5) ->dict:
    """
        对指定视频的关键帧进行 OCR 提取，并按时间窗口去重合并。

        返回字典格式: { window_index (int) : "该窗口内的 OCR 文本拼接" (str) }
        """
    keyframe_path = PROJECT_PATH / "storage" / "keyframes" / video_id
    if not keyframe_path.exists():
        logging.error(f"[{video_id}] Keyframes directory not found.")
        return {}
    window_text = {}

    #遍历所有关键帧
    for img_file in sorted(keyframe_path.glob("*.jpg")):
        try:
            #从文件名提取时间戳，stem用于返回去除扩展名的文件名
            timestamp_str = img_file.stem
            timestamp_str = float(timestamp_str)
            window_idx = int(timestamp_str // window_size)

            #ocr识别
            results = ocr_model.ocr(str(img_file))
            if not results or results[0] is None:
                continue
            res_dict = results[0]
            frame_text = []
            if "rec_texts" and "rec_scores" in res_dict:
                for text,score in zip(res_dict["rec_texts"],res_dict["rec_scores"]):
                    print(f"text : {text} ,socre:{score}")
                    if score > 0.85:
                        frame_text.append(text)

            if window_idx not in window_text:
                window_text[window_idx] = set()
            window_text[window_idx].update(frame_text)
        except Exception as e:
            logging.error(f"[{video_id}] OCR error on {img_file.name}: {str(e)}")



if __name__ == '__main__':
    test_id = "202abc3990af"
    process_ocr(test_id)


