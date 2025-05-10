#include "inference.h"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	try {
		Inference inf("yolov8n.onnx", cv::Size(640, 640));

		std::vector<std::string> imageNames;
		imageNames.push_back("bus.jpg");

		for (int i = 0; i < imageNames.size(); ++i)
		{
			cv::Mat frame = cv::imread(imageNames[i]);

			// Inference starts here...
			std::vector<Detection> output = inf.runInference(frame);

			int detections = output.size();
			std::cout << "Number of detections:" << detections << std::endl;

			for (int i = 0; i < detections; ++i)
			{
				Detection detection = output[i];

				cv::Rect box = detection.box;
				cv::Scalar color = detection.color;

				// Detection box
				cv::rectangle(frame, box, color, 2);

				// Detection box text
				std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
				cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
				cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

				cv::rectangle(frame, textBox, color, cv::FILLED);
				cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
			}
			// Inference ends here...

			// This is only for preview purposes
			float scale = 0.8;
			cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
			cv::imwrite("Inference.jpg", frame);
		}
	} catch (std::exception &E) {
		cout << "Error #97: std::exception in main: " << E.what() << "\n";
	} catch (...) {
		cout << "Error #102 in main. (No more infos.)\n";
	}

}
