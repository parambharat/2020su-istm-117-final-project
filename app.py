from fastapi import FastAPI
from src.pipeline import inference_pipeline

inference_config = inference_pipeline.Config()
inference_pipeline.InferencePipeline(inference_config)

app = FastAPI()


@app.post("/segment_text")
async def segment(text):
    segments, preprocessed = inference_pipeline.segment_text(text)
    return_obj = {
        "original_text": text,
        "preprocessed_text": preprocessed,
        "segments": segments,
    }
    return return_obj
