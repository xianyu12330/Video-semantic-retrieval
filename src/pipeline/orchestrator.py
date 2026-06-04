import asyncio
import logging
from src.pipeline.keyframe import extract_frame
async def process_video_pipeline(video_id:str,video_path:str,store_ref:dict):
    #视频处理流水线，帧的提取
    try:
        logging.info(f"[{video_id}] Pipeline started.")
        # 将CPU密集型的抽帧任务放到线程池执行，防止卡死 FastAPI
        frames = await asyncio.to_thread(extract_frame,video_path,video_id)
        logging.info(f"[{video_id}] Keyframe extraction completed. {len(frames)} frames kept.")

        #更新视频的状态
        store_ref[video_id]["status"] = "indexed"
        store_ref[video_id]["segment_count"] = len(frames)

    except Exception as e:
        logging.error(f"[{video_id}] Pipeline failed: {str(e)}")
        # 捕获异常，更新状态，防止前端一直 loading
        store_ref[video_id]["status"] = "failed"
        store_ref[video_id]["error_message"] = str(e)