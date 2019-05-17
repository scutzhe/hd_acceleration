#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <string>

namespace Tn
{
    const int INPUT_CHANNEL = 3;
    const std::string INPUT_PROTOTXT = "model/yolov3_416.prototxt";
    const std::string INPUT_CAFFEMODEL = "model/yolov3_416.caffemodel";
    //const std::string INPUT_IMAGE = "02.jpg";
    //const std::string INPUT_IMAGE ="";
    //const std::string EVAL_LIST = "";
    const std::string CALIBRATION_LIST = "";
    const std::string MODE = "int8";
    const std::string OUTPUTS= "yolo-det";
    const int INPUT_WIDTH = 416;
    const int INPUT_HEIGHT = 416;
    const int DETECT_CLASSES = 80;
    const float NMS_THRESH = 0.45;
}

#endif
