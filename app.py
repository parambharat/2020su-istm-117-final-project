from fastapi import FastAPI
from src.pipeline import inference_pipeline

inference_config = inference_pipeline.Config()
inference_pipe = inference_pipeline.InferencePipeline(inference_config)

app = FastAPI()


@app.post("/segment_text")
async def segment(req: dict):
    text = req["text"]
    segments, preprocessed = inference_pipe.segment_text(text)
    return_obj = {
        "original_text": text,
        "preprocessed_text": preprocessed,
        "segments": segments,
    }
    return return_obj
