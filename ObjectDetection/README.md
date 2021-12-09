For usage please follow https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html or https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api to install the Tensorflow 2 Object detection API

upgrade pip before installation

maybe this fix is needed:
https://github.com/tensorflow/models/issues/9706

numpy version 1.21.4 worked: pip install --upgrade numpy=1.21.4

protc error: https://stackoverflow.com/questions/45708443/tensorflow-object-detection-module-error-appear-when-trying-to-use-protoc

Update:
pip install mit --no-deps --upgrade