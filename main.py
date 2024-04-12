import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException


logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
LOGGER = logging.getLogger("openutau-remote-inference")
LOGGER.setLevel(logging.INFO)

ONNX_SESSIONS = {}
TORCHSCRIPT_MODELS = {}

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--root_dir", type=str, default=str(Path(__file__).parent), help="root directory containing models")
args = parser.parse_args()
args.root_dir = Path(args.root_dir).resolve()
LOGGER.info(f"Model root directory: {args.root_dir}")

app = FastAPI()

def process_path(filepath: str) -> Path:
    filepath = Path(filepath)
    if not filepath.is_absolute():
        filepath = args.root_dir / filepath
    real_filepath = os.path.realpath(filepath)
    real_root_dir = os.path.realpath(args.root_dir)
    if Path(os.path.commonprefix([real_filepath, real_root_dir])) != Path(real_root_dir):
        raise HTTPException(status_code=403, detail="Path provided is not a subpath of the root directory.")
    return filepath

@app.get("/ping")
async def ping() -> str:
    return "pong"

@app.get("/exists")
async def exists(model_path: str) -> bool:
    model_path = process_path(model_path)
    return model_path.exists()

@app.get("/onnx_info/inputs")
async def onnx_input_names(model_path: str) -> List[str]:
    model_path = process_path(model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model {model_path.relative_to(args.root_dir)} not found")
    if model_path not in ONNX_SESSIONS:
        LOGGER.warn(f"Model {model_path} not loaded, loading now")
        ONNX_SESSIONS[model_path] = ort.InferenceSession(
            model_path,
            providers=[
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        LOGGER.info(f"Model {model_path} loaded")
    session = ONNX_SESSIONS[model_path]
    return [input.name for input in session.get_inputs()]

@app.post("/release")
async def release(body: Dict):
    model_path = process_path(body['model_path'])
    if model_path in ONNX_SESSIONS:
        del ONNX_SESSIONS[model_path]
        LOGGER.info(f"Model {model_path.relative_to(args.root_dir)} released")
    if (model_path.parent / 'vocoder.yaml').exists():
        vocoder_config = yaml.safe_load((model_path.parent / 'vocoder.yaml').read_text())
        if vocoder_config.get('model_type', 'onnx') == 'jit':
            jit_model_path = model_path.with_suffix('.jit')
            if jit_model_path in TORCHSCRIPT_MODELS:
                del TORCHSCRIPT_MODELS[jit_model_path]
                LOGGER.info(f"Vocoder {model_path.with_suffix('.jit')} released")
    return "ok"

@app.post("/inference")
async def inference(body: Dict):
    model_path = process_path(body['model_path'])
    body_inputs = body['inputs']

    # Special case for DDSP vocoder
    if (model_path.parent / 'vocoder.yaml').exists():
        vocoder_config = yaml.safe_load((model_path.parent / 'vocoder.yaml').read_text())
        if vocoder_config.get('model_type', 'onnx') == 'jit':
            jit_model_path = model_path.with_suffix('.jit')
            if not jit_model_path.exists():
                raise HTTPException(status_code=404, detail=f"Torchscript model {jit_model_path.relative_to(args.root_dir)} not found")
            if jit_model_path not in TORCHSCRIPT_MODELS:
                TORCHSCRIPT_MODELS[jit_model_path] = torch.jit.load(model_path.with_suffix('.jit'), map_location=torch.device('cpu'))
                TORCHSCRIPT_MODELS[jit_model_path].eval()
                LOGGER.info(f"Vocoder {model_path.with_suffix('.jit')} loaded")
            vocoder_model = TORCHSCRIPT_MODELS[jit_model_path]
            for k in ['mel', 'f0']:
                if k not in body_inputs:
                    raise HTTPException(status_code=400, detail=f"Input {k} not found in request body")
                if body_inputs[k]['type'] != 'tensor(float)':
                    raise HTTPException(status_code=400, detail=f"Input {k} must be tensor(float)")
            mel_input, f0_input = body_inputs['mel'], body_inputs['f0']
            with torch.no_grad():
                signal, _, _ = vocoder_model(
                    torch.from_numpy(np.array(mel_input["float_data"], dtype=np.dtype("float32")).reshape(mel_input["shape"])),
                    torch.from_numpy(np.array(f0_input["float_data"], dtype=np.dtype("float32")).reshape(f0_input["shape"])).unsqueeze(-1)
                )
            return {
                'waveform': {
                    'type': 'tensor(float)',
                    'shape': signal.shape,
                    'float_data': signal.flatten().tolist(),
                }
            }

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model {model_path.relative_to(args.root_dir)} not found")
    if model_path not in ONNX_SESSIONS:
        LOGGER.warn(f"Model {model_path.relative_to(args.root_dir)} not loaded, loading now")
        ONNX_SESSIONS[model_path] = ort.InferenceSession(
            model_path,
            providers=[
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        LOGGER.info(f"Model {model_path.relative_to(args.root_dir)} loaded")
    session = ONNX_SESSIONS[model_path]

    inputs = {}
    for model_input in session.get_inputs():
        if model_input.name not in body_inputs:
            raise HTTPException(status_code=400, detail=f"Input {model_input.name} not found in request body")
        input = body_inputs[model_input.name]
        type_str = input["type"][7:-1]
        if (
            "type" not in input
            or "shape" not in input
            or f"{type_str}_data" not in input
        ):
            raise HTTPException(status_code=400, detail=f"Input {model_input.name} has invalid format")
        if model_input.type != input["type"]:
            raise HTTPException(
                status_code=400,
                detail=f"Input {model_input.name} has type {input['type']} but expected {model_input.type}"
            )
        inputs[model_input.name] = np.array(
            input[f"{type_str}_data"],
            dtype=np.dtype(type_str if type_str != "float" else "float32"),
        ).reshape(input["shape"])

    outputs = session.run(None, inputs)
    model_outputs = session.get_outputs()
    return {
        model_outputs[i].name: {
            "type": model_outputs[i].type,
            "shape": outputs[i].shape,
            f"{model_outputs[i].type[7:-1]}_data": outputs[i].flatten().tolist(),
        }
        for i in range(len(model_outputs))
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7889,
        workers=1,
        log_level="info",
        # reload=True,
    )
