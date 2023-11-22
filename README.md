# :satisfied: OpenUtau Remote Inference Server
This is a FastAPI server to delegate OpenUtau DiffSinger inference requests. For OpenUtau, you will need my forked version on `remote-inference` branch.

## :bookmark_tabs: Planned API
- `POST /onnx/:model_path` :heavy_check_mark:
- `POST /torchscript/:model_path` :large_orange_diamond:
- `POST /torch/:ckpt_path` :large_orange_diamond:

## :hamburger: Why
For one, I want to use DDSP vocoders in OpenUtau, and also try to use torch.compile directly on checkpoints.

## :bangbang: Disclaimer
This is not a packaged repo, including the modified OpenUtau, you are expected to prepare your own Python environment and build OpenUtau. Later I'll see what can be done.

### :yum: Thanks
Original idea ([https://github.com/fishaudio/openutau-remote-host](https://github.com/fishaudio/openutau-remote-host))
OpenUtau modification reference ([https://github.com/fishaudio/OpenUtau](https://github.com/fishaudio/OpenUtau))
