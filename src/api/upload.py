import uuid
import asyncio
from http.client import HTTPException
from pathlib import Path
from fastapi import FastAPI, File, UploadFile,BackgroundTasks
from src.pipeline.orchestrator import process_video_pipeline
# 使用项目根目录的绝对路径（假设 src/api 的上级目录是项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
STORAGE_PATH = PROJECT_ROOT / "src" / "storage"
app = FastAPI()
#内存中记录所有视频的状态
video_store : dict[str,dict] = {}
@app.post("/api/video/upload")
async def upload_video(file:UploadFile,background_tasks:BackgroundTasks = None):
    #create id
    video_id = uuid.uuid4().hex[:12]

    #check upload form
    if file.content_type not in ["video/mp4","video/avi","video/mov","video/x-matroska"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    # save to local
    type_name = file.filename.split(".",1)[-1]
    video_path = STORAGE_PATH /"videos"/f"{video_id}.{type_name}"
    video_path.parent.mkdir(parents=True,exist_ok=True)

    content = await file.read()
    if len(content) > 1024 * 1024 *1024:
        raise HTTPException(status_code=413, detail="File too large")

    video_path.write_bytes(content)

    #record status
    video_store[video_id] = {
        "video_id" :video_id,
        "video_name":file.filename,
        "status":"processing", #processing → indexed → failed/finished
        "duration":None,
        "segment_count":0,
        "error_message":None
    }

    #async start pipeline
    background_tasks.add_task(
        process_video_pipeline,
        video_id,
        str(video_path),
        video_store
    )
    return {"video_id":video_id,"video_name":file.filename}

#get video status
@app.get("/api/video/status/{video_id}")
async def get_status(video_id: str):
    if video_id not in video_store:
        raise HTTPException(status_code=404, detail="Video not found")
    return video_store[video_id]

if __name__ == '__main__':
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # 上传文件测试
    with open("G:\\ai Infra\\project\\src\\storage\\LimbusCompany .mp4", "rb") as f:
        response = client.post(
            "/api/video/upload",
            files={"file": ("dog_run.mp4", f, "video/mp4")}
        )

    print(response.status_code)
    print(response.json())

    # 获取状态测试
    if response.status_code == 200:
        video_id = response.json()["video_id"]
        status_response = client.get(f"/api/video/status/{video_id}")
        print(status_response.json())


