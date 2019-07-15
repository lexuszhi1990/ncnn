
### setup ncnn on machine

export VULKAN_SDK=/mnt/data-4t/workspace/david/vulkansdk_1.1.92.1/x86_64
cmake -DNCNN_VULKAN=OFF ..
cmake -DNCNN_VULKAN=ON ..

with VULKAN:
insert_inputs :5e-06 s
extract_out :0.02616 s
end :0.026177 s

532 = 0.136841
920 = 0.059326
598 = 0.049194

without VULKAN:
insert_inputs :4e-06 s
extract_out :0.140397 s
end :0.141593 s

532 = 0.136841
920 = 0.059326
598 = 0.049194

### convert pytorch to onnx to ncnn

docker run --rm -it -v /mnt/data-4t/workspace/david/ncnn:/app -w /app pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime-centernet bash

or

apt install -y cython libglib2.0-dev libsm6 libxext6 libxrender-dev
pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python onnx onnx-simplifier

python3 -m onnxsim squeezenet.onnx squeezenet-sim.onnx
onnx2ncnn squeezenet-sim.onnx squeezenet-torch.param squeezenet-torch.bin

### benchmark

gpu: ./benchncnn 8 2 0 0

          squeezenet  min =    1.88  max =    2.49  avg =    2.08
           mobilenet  min =    1.89  max =    2.47  avg =    2.18
        mobilenet_v2  min =    2.76  max =    3.61  avg =    3.13
          shufflenet  min =    2.38  max =    3.04  avg =    2.63
             mnasnet  min =    3.10  max =    3.75  avg =    3.24
     proxylessnasnet  min =    3.18  max =    3.46  avg =    3.23
           googlenet  min =    6.37  max =    6.40  avg =    6.39
            resnet18  min =    3.33  max =    3.39  avg =    3.35
             alexnet  min =    4.50  max =    4.56  avg =    4.54
               vgg16  min =   12.32  max =   12.71  avg =   12.48
            resnet50  min =    6.25  max =    6.61  avg =    6.40
      squeezenet_ssd  min =   10.40  max =   10.67  avg =   10.47
       mobilenet_ssd  min =    5.15  max =    6.00  avg =    5.42
      mobilenet_yolo  min =    5.96  max =    6.28  avg =    6.07
    mobilenet_yolov3  min =    5.11  max =    5.48  avg =    5.23


cpu: ./benchncnn 8 2 0

          squeezenet  min =   19.74  max =   19.90  avg =   19.81
     squeezenet_int8  min =   49.35  max =   51.19  avg =   49.81
           mobilenet  min =   35.41  max =   35.52  avg =   35.49
      mobilenet_int8  min =  101.43  max =  101.90  avg =  101.56
        mobilenet_v2  min =   29.47  max =   29.97  avg =   29.63
          shufflenet  min =   15.57  max =   15.61  avg =   15.59
             mnasnet  min =   42.04  max =   42.24  avg =   42.15
     proxylessnasnet  min =   57.15  max =   57.63  avg =   57.35
           googlenet  min =   76.66  max =   76.89  avg =   76.74
      googlenet_int8  min =  148.31  max =  148.59  avg =  148.41
            resnet18  min =   59.94  max =   61.01  avg =   60.29
       resnet18_int8  min =  100.82  max =  101.29  avg =  100.97
             alexnet  min =   82.14  max =   84.11  avg =   82.83
               vgg16  min =  235.07  max =  239.38  avg =  238.09
          vgg16_int8  min =  442.41  max =  449.91  avg =  445.63
            resnet50  min =  148.47  max =  155.56  avg =  151.01
       resnet50_int8  min =  391.22  max =  395.82  avg =  391.86
      squeezenet_ssd  min =   47.17  max =   47.41  avg =   47.26
 squeezenet_ssd_int8  min =   75.36  max =   76.31  avg =   75.78
       mobilenet_ssd  min =   66.84  max =   67.29  avg =   66.94
  mobilenet_ssd_int8  min =  202.25  max =  223.75  avg =  205.58
      mobilenet_yolo  min =  151.86  max =  153.26  avg =  152.53
    mobilenet_yolov3  min =  157.07  max =  157.47  avg =  157.24

### build android

cmake -DCMAKE_TOOLCHAIN_FILE=/mnt/data-4t/Android/SDK/ndk-bundle/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON     -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON  ..

### build android-gpu

export VULKAN_SDK=/mnt/data-4t/workspace/david/vulkansdk_1.1.92.1/x86_64


reference
--------

https://github.com/daquexian/onnx-simplifier
