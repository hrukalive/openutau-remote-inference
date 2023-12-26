import logging
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

from nvSTFT import STFT

LOGGER = logging.getLogger("openutau-remote-inference")
ONNX_SESSIONS = {}
TORCHSCRIPT_MODELS = {}

app = FastAPI()

@app.post("/inference/{model_path}")
def inference(model_path: Path, body: Dict):
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

    inputs = {}
    for model_input in session.get_inputs():
        if model_input.name not in body:
            raise Exception(f"Input {model_input.name} not found in request body")
        input = body[model_input.name]
        type_str = input["type"][7:-1]
        if (
            "type" not in input
            or "shape" not in input
            or f"{type_str}_data" not in input
        ):
            raise Exception(f"Input {model_input.name} has invalid format")
        if model_input.type != input["type"]:
            raise Exception(
                f"Input {model_input.name} has type {input['type']} but expected {model_input.type}"
            )
        inputs[model_input.name] = np.array(
            input[f"{type_str}_data"],
            dtype=np.dtype(type_str if type_str != "float" else "float32"),
        ).reshape(input["shape"])

    if (model_path.parent / 'vocoder.yaml').exists() and model_path.with_suffix('.jit').exists():
        vocoder_config = yaml.safe_load((model_path.parent / 'vocoder.yaml').read_text())
        if vocoder_config.get('model_type', 'onnx') in ['jit', 'combined']:
            jit_model_path = model_path.with_suffix('.jit')
            if jit_model_path not in TORCHSCRIPT_MODELS:
                TORCHSCRIPT_MODELS[jit_model_path] = torch.jit.load(model_path.with_suffix('.jit'), map_location=torch.device('cpu'))
                TORCHSCRIPT_MODELS[jit_model_path].eval()
                LOGGER.info(f"Vocoder {model_path.with_suffix('.jit')} loaded")
            vocoder_model = TORCHSCRIPT_MODELS[jit_model_path]
            with torch.no_grad():
                signal, _, _ = vocoder_model(torch.from_numpy(inputs['mel']), torch.from_numpy(inputs['f0']).unsqueeze(-1))
            if vocoder_config.get('model_type', 'onnx') == 'jit':
                return {
                    'waveform': {
                        'type': 'tensor(float)',
                        'shape': signal.shape,
                        'float_data': signal.flatten().tolist(),
                    }
                }
            else:
                try:
                    stft = STFT(
                        vocoder_config['sample_rate'],
                        vocoder_config['num_mel_bins'],
                        vocoder_config['n_fft'],
                        vocoder_config['win_length'],
                        vocoder_config['hop_size'],
                        vocoder_config['mel_fmin'],
                        vocoder_config['mel_fmax'],
                    )
                except KeyError as e:
                    raise HTTPException(status_code=500, detail=f"Vocoder config file {model_path.parent / 'vocoder.yaml'} is missing key {e}")
                new_mel = stft.get_mel(signal).cpu().numpy()
                inputs['mel'] = new_mel.transpose(0, 2, 1)

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
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    LOGGER.setLevel(logging.INFO)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7889,
        workers=1,
        log_level="info",
        # reload=True,
    )
