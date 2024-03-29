#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace std;


int main() {

	vector<cv::Mat> tramp;
	for (int i = 0; i < 13; i++) {
		std::stringstream sst;
		sst << "./img/s" << i + 1 << ".jpg";
		tramp.push_back(cv::imread(sst.str(), 0));
	}

	const int cols = 256;
	const int rows = (int)(cols * 89.0/58.0);

	//camera parameters
	double fx = 525.0; //focal
	double fy = 525.0; //focal
	double cx = 319.5; //optical centre x
	double cy = 239.5; //optical centre y

	cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
	cameraMatrix.at<double>(0, 0) = fx;
	cameraMatrix.at<double>(1, 1) = fy;
	cameraMatrix.at<double>(2, 2) = 1;
	cameraMatrix.at<double>(0, 2) = cx;
	cameraMatrix.at<double>(1, 2) = cy;
	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(1, 0) = 0;
	cameraMatrix.at<double>(2, 0) = 0;
	cameraMatrix.at<double>(2, 1) = 0;

	cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
	distCoeffs.at<double>(0) = 0;
	distCoeffs.at<double>(1) = 0;
	distCoeffs.at<double>(2) = 0;
	distCoeffs.at<double>(3) = 0;

	cv::VideoCapture cap(1);
	int Width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int Height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	fx = fy = (double)Width;

	bool loopFlag = true;
	bool binFlag = false;
	bool edgeFlag = false;
	bool polyFlag = false;
	bool trampFlag = false;
	bool labelFlag = false;
	int xorNum = 0;
	bool cubeFlag = false;



	// esc を押すまで
	while (loopFlag) {
		cv::Mat frame;
		cap >> frame;
		// キャプチャできていなければ処理を飛ばす
		if (!frame.data) {
			continue;
		}

		// 2値化
		cv::Mat grayImage, binImage;
		cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
		cv::threshold(grayImage, binImage, 128.0, 255.0, cv::THRESH_OTSU);
		if (binFlag) cv::imshow("bin", binImage);
		else cv::destroyWindow("bin");

		// 輪郭抽出
		std::vector<std::vector< cv::Point >> contours;
		cv::findContours(binImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		// 検出された輪郭線の描画
		if (edgeFlag) {
			for (auto contour = contours.begin(); contour != contours.end(); contour++) {
				cv::polylines(frame, *contour, true, cv::Scalar(0, 255, 0), 2);
			}
		}

		// 輪郭が四角形かの判定
		for (auto contour = contours.begin(); contour != contours.end(); contour++) {
			// 輪郭を直線近似
			std::vector< cv::Point > approx;
			cv::approxPolyDP(cv::Mat(*contour), approx, 50.0, true);
			// 近似が4線かつ面積が一定以上なら四角形
			double area = cv::contourArea(approx);
			if (approx.size() == 4 && area > 100.0) {
				cv::Mat dst(rows, cols, CV_64FC4);
				cv::Point2f dstPoints[4];

				float da = std::min(cv::norm(approx[0] - approx[1]), cv::norm(approx[2] - approx[3]));
				float db = std::min(cv::norm(approx[1] - approx[2]), cv::norm(approx[3] - approx[0]));

				if (da < db) {
					approx.push_back(approx[0]);
					approx.erase(approx.begin());
				}

				dstPoints[0] = { 0,				0 };
				dstPoints[1] = { 0,				(float)rows };
				dstPoints[2] = { (float)cols,	(float)rows };
				dstPoints[3] = { (float)cols,	0 };
				cv::Point2f aprPoints[4]{ approx[0], approx[1], approx[2], approx[3] };
				if (polyFlag) cv::polylines(frame, approx, true, cv::Scalar(255, 0, 0), 2);

				cv::Mat H = getPerspectiveTransform(aprPoints, dstPoints);
				warpPerspective(binImage, dst, H, dst.size());

				cv::Mat img_xor;
				float score = 255;
				int num = 0;
				for (int i = 0; i < 13; i++) {
					bitwise_xor(dst, tramp[i], img_xor);
					float mean = cv::mean(img_xor)[0];
					if (mean < score) {
						score = mean;
						num = i + 1;
					}
				}
				if (xorNum > 0) {
					bitwise_xor(dst, tramp[xorNum - 1], img_xor);
					cv::imshow("xor", img_xor);
				} else {
					cv::destroyWindow("xor");
				}
				
				if (score < 10) {
					if (labelFlag) cv::putText(frame, std::to_string(num), (approx[0] + approx[2]) / 2 + cv::Point{ -5, 5 }, cv::FONT_HERSHEY_PLAIN, 6.0, cv::Scalar(0, 0, 255), 3);

					// solve PnP
					cv::Mat rvec(3, 1, cv::DataType<double>::type);
					cv::Mat tvec(3, 1, cv::DataType<double>::type);

					std::vector<cv::Point3f > objectPoints;
					objectPoints.push_back(cv::Point3f(0, 0, 10));
					objectPoints.push_back(cv::Point3f(0, (float)rows, 10));
					objectPoints.push_back(cv::Point3f((float)cols, (float)rows, 10));
					objectPoints.push_back(cv::Point3f((float)cols, 0, 10));

					std::vector<cv::Point2f> imagePoints;
					imagePoints.push_back((cv::Point2f)approx[0]);
					imagePoints.push_back((cv::Point2f)approx[1]);
					imagePoints.push_back((cv::Point2f)approx[2]);
					imagePoints.push_back((cv::Point2f)approx[3]);

					cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

					float h = 100.0f;
					float w = (float)cols;
					float l = (float)rows;
					std::vector<cv::Point3f> cube;
					cube.push_back(cv::Point3f(0, 0, 0));
					cube.push_back(cv::Point3f(w, 0, 0));
					cube.push_back(cv::Point3f(w, l, 0));
					cube.push_back(cv::Point3f(0, l, 0));
					cube.push_back(cv::Point3f(0, 0, -h));
					cube.push_back(cv::Point3f(w, 0, -h));
					cube.push_back(cv::Point3f(w, l, -h));
					cube.push_back(cv::Point3f(0, l, -h));


					if (cubeFlag) {
						std::vector<cv::Point2f> cube2D;
						cv::projectPoints(cube, rvec, tvec, cameraMatrix, distCoeffs, cube2D);
						cv::line(frame, cube2D[0], cube2D[1], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[1], cube2D[2], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[2], cube2D[3], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[3], cube2D[0], cv::Scalar(255, 255, 255), 1, 4);

						cv::line(frame, cube2D[4], cube2D[5], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[5], cube2D[6], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[6], cube2D[7], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[7], cube2D[4], cv::Scalar(255, 255, 255), 1, 4);

						cv::line(frame, cube2D[0], cube2D[4], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[1], cube2D[5], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[2], cube2D[6], cv::Scalar(255, 255, 255), 1, 4);
						cv::line(frame, cube2D[3], cube2D[7], cv::Scalar(255, 255, 255), 1, 4);
					}
				}

				if (trampFlag) cv::imshow("dst", dst);
				else cv::destroyWindow("dst");

				switch (cv::waitKey(10)) {
					case '1': binFlag = !binFlag; break;
					case '2': edgeFlag = !edgeFlag; break;
					case '3': polyFlag = !polyFlag; break;
					case '4': trampFlag = !trampFlag; break;
					case '5': xorNum = (xorNum + 1) % 14; break;
					case '6': labelFlag = !labelFlag; break;
					case '7': cubeFlag = !cubeFlag; break;
					case 's': cv::imwrite("./img/test.jpg", frame); break;
					case 0x1b:
						loopFlag = false;
						break;
				}
			}
		}
		cv::imshow("frame", frame);
	}
}

