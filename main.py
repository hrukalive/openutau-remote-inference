from fastapi import FastAPI
from typing import Dict
import logging
import uvicorn
from threading import Lock
import onnxruntime as ort
import numpy as np

LOGGER = logging.getLogger('openutau-remote-inference')

app = FastAPI()

ONNX_SESSIONS = {}
ONNX_LOCK = Lock()

@app.post("/onnx/{model_path}")
async def onnx_inference(model_path, body: Dict):
    global ONNX_SESSIONS, ONNX_LOCK
    with ONNX_LOCK:
        if model_path not in ONNX_SESSIONS:
            LOGGER.warn(f"Model {model_path} not loaded, loading now")
            ONNX_SESSIONS[model_path] = ort.InferenceSession(
                model_path, providers=[
                    # 'TensorrtExecutionProvider',
                    # 'CUDAExecutionProvider',
                    'DmlExecutionProvider',
                    'CPUExecutionProvider',
                ]
            )
        session = ONNX_SESSIONS[model_path]
        LOGGER.info(f"Model {model_path} loaded")

        inputs = {}
        for model_input in session.get_inputs():
            if model_input.name not in body:
                raise Exception(f"Input {model_input.name} not found in request body")
            input = body[model_input.name]
            type_str = input['type'][7:-1]
            if 'type' not in input or 'shape' not in input or f'{type_str}_data' not in input:
                raise Exception(f"Input {model_input.name} has invalid format")
            if model_input.type != input['type']:
                raise Exception(f"Input {model_input.name} has type {input['type']} but expected {model_input.type}")
            inputs[model_input.name] = np.array(input[f'{type_str}_data'], dtype=np.dtype(type_str if type_str != 'float' else 'float32')).reshape(input['shape'])

    outputs = session.run(None, inputs)
    model_outputs = session.get_outputs()
    return {
        model_outputs[i].name: {
            'type': model_outputs[i].type,
            'shape': outputs[i].shape,
            f"{model_outputs[i].type[7:-1]}_data": outputs[i].flatten().tolist(),
        } for i in range(len(model_outputs))
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
    LOGGER.setLevel(logging.INFO)

    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=7889,
        workers=4,
        log_level='info',
        reload=True,
    )
