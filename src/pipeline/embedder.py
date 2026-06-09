import os
import numpy as np
import torch.cuda
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional, Dict
from pathlib import Path
PROJECT_PATH = Path(__file__).parent.parent
MODEL_PATH = PROJECT_PATH.parent / "BAAI" / "bge-m3"

class TextEmbedder:
    def __init__(self,mode_path:str = str(MODEL_PATH)):
        self.model_path = mode_path
        self.device = "cuda"
        self.model = None

    def load(self):
        if self.model is None:
            logging.info(f"Loading BGE-M3 model to {self.device}...")
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device
            )
        return self

    def unload(self):
        if self.model is not None:
            logging.info("Unloading BGE-M3 model and clearing VRAM...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def encode(self,text:list[str]) ->Optional[np.ndarray]:
        """
                核心模型推理逻辑。
                接收文本列表，利用 SentenceTransformer 的批处理能力进行推理。
                """
        if text is None:
            return None
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        sentences = text
        embedding = self.model.encode(
            sentences,
            batch_size=16,
            normalize_embeddings=True,

        )
        return embedding

    def process_window_dict(self,window_text:Dict[int,str])->Dict[int,np.ndarray]:
        window_array = {}
        for idx,text in window_text.items():
            if text is None:
                continue
            if idx not in window_array:
                window_array[idx] = self.encode([text])
        return window_array


# if __name__ == '__main__':



