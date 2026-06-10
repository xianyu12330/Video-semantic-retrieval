import asyncio
import logging
from typing import List, Optional, Dict
from src.pipeline.keyframe import extract_frame
from src.pipeline.ocr import process_ocr
from src.pipeline.asr import asr_extract
from src.pipeline.embedder import TextEmbedder

async def process_video_pipeline(video_id:str,video_path:str,store_ref:dict):
    #视频处理流水线，帧的提取
    try:
        logging.info(f"[{video_id}] Pipeline started.")
        # 将CPU密集型的抽帧任务放到线程池执行，防止卡死 FastAPI
        frames = await asyncio.to_thread(extract_frame,video_path,video_id)
        logging.info(f"[{video_id}] Keyframe extraction completed. {len(frames)} frames kept.")



        # 2. 并发执行 ASR 和 OCR 提取 (同样交由线程池)
        ocr_result,asr_result = await asyncio.gather(
            asyncio.to_thread(process_ocr,video_id),
            asyncio.to_thread(asr_extract,video_id)
        )
        # 3. 串行执行 Embedding 推理
        embedder = TextEmbedder()
        try:
            embedder.load()
            #使用同一个实例，串行推理
            ocr_embedding = await asyncio.to_thread(embedder.process_window_dict, ocr_result)
            asr_embedding = await asyncio.to_thread(embedder.process_window_dict, asr_result)
        finally:
            embedder.unload()
        # 4. （准备对接 Milvus）
        # TODO: 将 frames, ocr_embedding, asr_embedding 时序对齐并写入 Milvus
        # 更新视频的状态
        store_ref[video_id]["status"] = "indexed"
        store_ref[video_id]["segment_count"] = len(frames)

    except Exception as e:
        logging.error(f"[{video_id}] Pipeline failed: {str(e)}")
        # 捕获异常，更新状态，防止前端一直 loading
        store_ref[video_id]["status"] = "failed"
        store_ref[video_id]["error_message"] = str(e)


