# hd_acceleration
# TRTForYolov3

## Desc

    tensorRT for Yolov3
	
### Test Enviroments

    Ubuntu  16.04
    TensorRT 5.0.2.6
    CUDA 9.0
    python 3.5

### Models

Download the caffe model converted by official model:

+ model [here](https://pan.baidu.com/s/1cVIQcAEsZO5p4S-vGf4CYQ) pwd:sqpr

### Run Sample

```bash
#build source code
git submodule update --init --recursive
mkdir build
cd build && cmake .. && make && make install
cd ..
```
#for yolov3
```
> sh demo.sh
```
### Performance

### Details About Wrapper

see link [TensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper)
