import uuid
import asyncio
from pathlib import Path

# 内存中记录所有视频的状态（重启丢失，够用）
video_store: dict[str, dict] = {}


@router.post("/api/videos/upload")
async def upload_video(file: UploadFile):
    # 1. 生成唯一ID
    video_id = uuid.uuid4().hex[:12]

    # 2. 校验格式
    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/x-matroska"]:
        raise HTTPException(400, "不支持的格式，仅接受 mp4/avi/mov/mkv")

    # 3. 保存到磁盘
    ext = file.filename.rsplit(".", 1)[-1]
    video_path = Path(f"data/videos/{video_id}.{ext}")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:  # 500MB限制
        raise HTTPException(413, "文件超过1G")

    video_path.write_bytes(content)

    # 4. 记录状态
    video_store[video_id] = {
        "video_id": video_id,
        "video_name": file.filename,
        "status": "processing",      # processing → indexed → failed
        "file_path": str(video_path),
        "duration": None,
        "segment_count": 0,
        "error_message": None,
    }

    # 5. 异步启动 Pipeline（不等它跑完，立刻返回）
    asyncio.create_task(run_pipeline(video_id))

    # 6. 立即返回
    return {"video_id": video_id, "video_name": file.filename, "status": "processing"}


# GET /api/videos/{video_id}/status
@router.get("/api/videos/{video_id}/status")
async def get_status(video_id: str):
    if video_id not in video_store:
        raise HTTPException(404, "视频不存在")
    return video_store[video_id]
