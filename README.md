# :satisfied: OpenUtau Remote Inference Server
This is a FastAPI server to delegate OpenUtau DiffSinger inference requests. For OpenUtau, you will need [my modified OpenUtau](https://github.com/hrukalive/OpenUtau/tree/remote-inference).

To start the server: `python main.py [-d <MODEL_ROOT_DIR (default to .)>]`

OpenUtau will send relative model paths (ex. `Singers/Kiritan/acoustic.onnx` or `Singers/Kiritan/dspitch/pitch.onnx`) to the server for inference, and the server will search for the model file under its `MODEL_ROOT_DIR`. So you can replicate the file structure of the `Singers` folder and move all models to the remote server, and leave other files in OpenUtau folder. (You can also keep a copy in OpenUtau because when the remote server is unreachable, it will fall back to local inference).

#### :information_desk_person: DDSP Vocoder

You can also copy `vocoder.yaml` to the remote server alongside with the finetuned HiFiGAN model. But the server also accepts DDSP JIT model. The trick is to name the JIT file as specified in the `vocoder.yaml` with `.jit` suffix, and add a new line `model_type: jit` to the YAML file. Now the server uses DDSP Vocoder!

## :bookmark_tabs: Planned API
- `GET /exists?model_path={model_path}` :heavy_check_mark:
- `GET /onnx_info/inputs?model_path={model_path}` :heavy_check_mark:
- `POST /inference` :heavy_check_mark:

## :hamburger: Why
For one, I want to use DDSP vocoders in OpenUtau, and also try to use torch.compile directly on checkpoints.

## :bangbang: Disclaimer
This is not a packaged repo, including [my modified OpenUtau](https://github.com/hrukalive/OpenUtau/tree/remote-inference), you are expected to prepare your own Python environment and build OpenUtau. Later I'll see what can be done.

### :yum: Thanks
- Original idea ([https://github.com/fishaudio/openutau-remote-host](https://github.com/fishaudio/openutau-remote-host))
- OpenUtau modification reference ([https://github.com/fishaudio/OpenUtau](https://github.com/fishaudio/OpenUtau))
