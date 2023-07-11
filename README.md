# Introduction

# Use
- ubuntu 22.04
- python tensorflow_classification.py
- create model.tflite
- python rknn_transfer.py
- create model.rknn
- debain 10 in rk3568
- https://github.com/rockchip-linux/rknpu2/blob/master/runtime/RK356X/Linux/librknn_api/aarch64/librknnrt.so
- scp librknnrt.so /usr/lib
- https://github.com/rockchip-linux/rknn-toolkit2/blob/master/rknn_toolkit_lite2/packages/rknn_toolkit_lite2-1.5.0-cp37-cp37m-linux_aarch64.whl
- pip install *.whl
- python rklite_ts.py

