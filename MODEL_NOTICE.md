# Model notice

SignBridge uses an external pretrained ASL alphabet recognition model:

Repository:
https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition

The referenced project describes its model as a 26-class ASL fingerspelling model and provides the ONNX weights used by this application.

The model is downloaded by `app.py` on first run from the original public repository and cached locally.

The original repository states that its code is MIT licensed but also notes that the model weights were trained using third-party datasets with their own terms. Review the upstream repository and dataset/model terms before redistributing the weights or using the project commercially.

SignBridge itself does not claim ownership of the upstream model weights.
