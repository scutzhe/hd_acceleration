#include <opencv2/opencv.hpp> //python C++ 混合编程
#include <chrono>
#include <vector>
#include <typeinfo>
#include <string>
#include <ctime>
//#include </usr/include/python3.6m/Python.h>
//#include <numpy/arrayobject.h> //包含 numpy 中的头文件arrayobject.h
#include "argsParser.h"
#include "configs.h"
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"
#include "TrtNet.h"

#ifdef __cplusplus
extern "C"{ //在extern “C”中的函数才能被外部调用
#endif

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace Yolo;
using namespace cv;


typedef struct StructPointerTest
{
    int num;
	int location[400];
}StructPointerTest, *StructPointer;


vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void DoNms(vector<Detection>& detections,int classes ,float nmsThresh)
{
    auto t_start = chrono::high_resolution_clock::now();

    vector<vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i];
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
}


vector<Bbox> postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes)
{
    using namespace cv;

    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    //nms
    float nmsThresh = parser::getFloatValue("nms");
    if(nmsThresh > 0)
        DoNms(detections,classes,nmsThresh);

    vector<Bbox> boxes;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        Bbox bbox =
        {
            item.classId,   //classId
            max(int((b[0]-b[2]/2.)*width),0), //left
            min(int((b[0]+b[2]/2.)*width),width), //right
            max(int((b[1]-b[3]/2.)*height),0), //top
            min(int((b[1]+b[3]/2.)*height),height), //bot
            item.prob       //score
        };
        boxes.push_back(bbox);
    }

    return boxes;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}


//int test(int *matrix, int rows, int columns, int channel)
int test(unsigned char *matrix, int width, int height, int channel)
{
    printf("int test func");
    if (channel == 3)
    {
        cv::Mat img(width, height, CV_8UC3, matrix);
        //cv::imwrite("result.jpg", img);
    }
    else if (channel == 1)
    {
        cv::Mat img(width, height, CV_8U, matrix);
        //cv::imwrite("result.jpg", img);
    }
    return 1;
}


StructPointer yolov3(unsigned char *matrix, int width, int height, int channel)
{
    StructPointer p = (StructPointer)malloc(sizeof(StructPointerTest));
    cv::Mat img_zhengxiangzhong;
    if (channel == 3)
    {
        cv::Mat img(width, height, CV_8UC3, matrix);
        img_zhengxiangzhong = img;
    }
    else if (channel == 1)
    {
        cv::Mat img(width, height, CV_8U, matrix);
        img_zhengxiangzhong = img;
    }

    using namespace cv;
    parser::ADD_ARG_STRING("prototxt",Desc("input yolov3 deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_STRING("caffemodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("calib",Desc("calibration image List"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("int8"));
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));
    parser::ADD_ARG_FLOAT("nms",Desc("non-maximum suppression value"),DefaultValue(to_string(NMS_THRESH)));

    string deployFile = parser::getStringValue("prototxt");
    string caffemodelFile = parser::getStringValue("caffemodel");

    vector<vector<float>> calibData;
    string calibFileList = parser::getStringValue("calib");
    string mode = parser::getStringValue("mode");

    if(calibFileList.length() > 0 && mode == "int8")
    {
        cout << "find calibration file,loading ..." << endl;

        ifstream file(calibFileList);
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << calibFileList << endl;
            exit(-1);
        }

        string strLine;
        while( getline(file,strLine) )
        {
            cv::Mat img = cv::imread(strLine);
            auto data = prepareImage(img);
            calibData.emplace_back(data);
        }
        file.close();
    }

    RUN_MODE run_mode = RUN_MODE::FLOAT32;
    if(mode == "int8")
    {
        if(calibFileList.length() == 0)
                cout<<"";
        else
            run_mode = RUN_MODE::INT8;
    }
    else if(mode == "fp16")
    {
        run_mode = RUN_MODE::FLOAT16;
    }

    string outputNodes = parser::getStringValue("outputs");
    auto outputNames = split(outputNodes,',');

    //can load from file
    string saveName = "yolov3_" + mode + ".engine";
    static trtNet net2(saveName);
    trtNet *net = &net2;

    int outputCount = net->getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);

    string listFile = parser::getStringValue("evallist");
    list<vector<Bbox>> groundTruth;

    list<vector<Bbox>> outputs;
    int classNum = parser::getIntValue("class");

    //std::cout << "process: starting" <<"......"<< std::endl;

    // read img_path
    vector<float> inputData = prepareImage(img_zhengxiangzhong);

    net->doInference(inputData.data(), outputData.get());

    //cout<<"time inference = "<<endTime-startTime<<endl;
    //Get Output
    auto output = outputData.get();

    //first detect count
    int count = output[0];

    //later detect result
    vector<Detection> result;
    result.resize(count);
    memcpy(result.data(), &output[1], count*sizeof(Detection));

    auto boxes = postProcessImg(img_zhengxiangzhong,result,classNum);
    outputs.emplace_back(boxes);

    //net.printTime();

    if(groundTruth.size() > 0)
    {
        //eval map
        evalMAPResult(outputs,groundTruth,classNum,0.5f);
        evalMAPResult(outputs,groundTruth,classNum,0.75f);
    }

    vector<int> coordinations;

    auto bbox = *outputs.begin();
    int index = 0;
    for(const auto& item : bbox)
    {
        if(item.classId !=0)
            continue;
        // cout <<item.left << "," <<item.top << ","<<item.right << ","<<item.bot << "\n";
        p->location[index++] = item.left;
    	p->location[index++] = item.top;
    	p->location[index++] = item.right;
    	p->location[index++] = item.bot;
    }
    p->num = index / 4;

    /*
    int j = 0;
    for(int i =0;i<coordinations.size();i++)
    {
        j = j+1;
        cout << "i = "<<coordinations[i] << ",";
        if(j%4==0)
            cout <<endl;
        else
            continue;
    }
    */
    return p;

}

#ifdef __cplusplus
}//匹配extern “C”中大括号 完成整个匹配
#endif