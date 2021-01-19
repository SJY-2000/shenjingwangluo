#include <iostream>
#include <opencv2/opencv.hpp>
#include <dnn.hpp>

#define DEMO_METHOD 1      
#define YOLOV3_VIDEO "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\\\openposeTest.mp4"      
#define OPENPOSE_VIDEO "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\\\openposeTest.mp4"    

using namespace std;
using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

float confThreshold = 0.5; 
float nmsThreshold = 0.4;  
int inpWidth = 416;       
int inpHeight = 416;       
int POSE_PAIRS[3][20][2] = {
	{ 
		{ 1,2 },{ 1,5 },{ 2,3 },
		{ 3,4 },{ 5,6 },{ 6,7 },
		{ 1,8 },{ 8,9 },{ 9,10 },
		{ 1,11 },{ 11,12 },{ 12,13 },
		{ 1,0 },{ 0,14 },
		{ 14,16 },{ 0,15 },{ 15,17 }
	},
	{   
		{ 0,1 },{ 1,2 },{ 2,3 },
		{ 3,4 },{ 1,5 },{ 5,6 },
		{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
		{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
	},
	{   
		{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         
		{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         
		{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },   
		{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  
		{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }  
	} };

void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs);
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
int yoloV3();
int openpose();

int main()
{
	
	double start = static_cast<double>(cvGetTickCount());

	int method = DEMO_METHOD;

	if (method == 0) {
		yoloV3();
	}
	else if (method == 1) {
		openpose();
	}

	double time = ((double)cvGetTickCount() - start) / cvGetTickFrequency();
	
	cout << "processing time:" << time / 1000 << "ms" << endl;

	system("pause");
	return 0;
}

std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		std::vector<cv::String> layersNames = net.getLayerNames();

		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs)
{
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point classIdPoint;
			double confidence;

			cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255));

	std::string label = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}
	else
	{
		std::cout << "classes is empty..." << std::endl;
	}

	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
	top = std::max(top, labelSize.height);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

int yoloV3()
{

	VideoCapture cap(YOLOV3_VIDEO);

	if (!cap.isOpened())return -1;

	string classesFile = "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\coco.names";
	
	String yolov3_model = "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\yolov3.cfg";
	
	String weights = "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\yolov3.weights";

	std::ifstream classNamesFile(classesFile.c_str());
	if (classNamesFile.is_open())
	{
		std::string className = "";
		
		while (std::getline(classNamesFile, className)) {
			classes.push_back(className);
		}
	}
	else {
		std::cout << "can not open classNamesFile" << std::endl;
	}

	cv::dnn::Net net = cv::dnn::readNetFromDarknet(yolov3_model, weights);

	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);

	cv::Mat frame;

	while (1)
	{
		cap >> frame;

		if (frame.empty()) {
			std::cout << "frame is empty!!!" << std::endl;
			return -1;
		}

		cv::Mat blob;
		cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);

		net.setInput(blob);

		std::vector<cv::Mat> outs;
	
		net.forward(outs, getOutputsNames(net));

		postprocess(frame, outs);

		cv::imshow("frame", frame);

		if (cv::waitKey(10) == 27)
		{
			break;
		}
	}

	return 0;
}

int openpose()
{

	String modelTxt = "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\openpose_pose_coco.prototxt";
	String modelBin = "D:\\vs2015_code\\week15_1\\DeepNeuralNetwork\\pose_iter_440000.caffemodel";

	cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);

	int W_in = 368;
	int H_in = 368;
	float thresh = 0.1;

	VideoCapture cap;
	cap.open(OPENPOSE_VIDEO);

	if (!cap.isOpened())return -1;

	while (1) {

		cv::Mat frame;

		cap >> frame;

		if (frame.empty()) {
			std::cout << "frame is empty!!!" << std::endl;
			return -1;
		}
		Mat inputBlob = blobFromImage(frame, 1.0 / 255, Size(W_in, H_in), Scalar(0, 0, 0), false, false);

		net.setInput(inputBlob);

		Mat result = net.forward();

		int midx, npairs;
		int H = result.size[2];
		int W = result.size[3];

		int nparts = result.size[1];

		if (nparts == 19)
		{   
			midx = 0;
			npairs = 17;
			nparts = 18; 
		}
		else if (nparts == 16)
		{  
			midx = 1;
			npairs = 14;
		}
		else if (nparts == 22)
		{  
			midx = 2;
			npairs = 20;
		}
		else
		{
			cerr << "there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has " << nparts << " parts." << endl;
			return (0);
		}

		vector<Point> points(22);
		for (int n = 0; n < nparts; n++)
		{
			Mat heatMap(H, W, CV_32F, result.ptr(0, n));
			
			Point p(-1, -1), pm;
			double conf;
			minMaxLoc(heatMap, 0, &conf, 0, &pm);
		
			if (conf > thresh) {
				p = pm;
			}
			points[n] = p;
		}

		float SX = float(frame.cols) / W;
		float SY = float(frame.rows) / H;
		for (int n = 0; n < npairs; n++)
		{
			Point2f a = points[POSE_PAIRS[midx][n][0]];
			Point2f b = points[POSE_PAIRS[midx][n][1]];

			if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
				continue;

			a.x *= SX; a.y *= SY;
			b.x *= SX; b.y *= SY;

			line(frame, a, b, Scalar(0, 200, 0), 2);
			circle(frame, a, 3, Scalar(0, 0, 200), -1);
			circle(frame, b, 3, Scalar(0, 0, 200), -1);
		}

		imshow("frame", frame);

		waitKey(30);

	}

	return 0;
}