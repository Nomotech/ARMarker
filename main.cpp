#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace std;


int main() {

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

	// esc ÇâüÇ∑Ç‹Ç≈
	while (cv::waitKey(5) != 0x1b) {
		cv::Mat frame;
		cap >> frame;
		// ÉLÉÉÉvÉ`ÉÉÇ≈Ç´ÇƒÇ¢Ç»ÇØÇÍÇŒèàóùÇîÚÇŒÇ∑
		if (!frame.data) {
			continue;
		}

		// 2ílâª
		cv::Mat grayImage, binImage;
		cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY);
		cv::threshold(grayImage, binImage, 128.0, 255.0, cv::THRESH_OTSU);
		cv::imshow("bin", binImage);

		// ó÷äsíäèo
		std::vector< std::vector< cv::Point > > contours;
		cv::findContours(binImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		// åüèoÇ≥ÇÍÇΩó÷äsê¸ÇÃï`âÊ
		for (auto contour = contours.begin(); contour != contours.end(); contour++) {
			cv::polylines(frame, *contour, true, cv::Scalar(0, 255, 0), 2);
		}

		// ó÷äsÇ™éläpå`Ç©ÇÃîªíË
		for (auto contour = contours.begin(); contour != contours.end(); contour++) {
			// ó÷äsÇíºê¸ãﬂéó
			std::vector< cv::Point > approx;
			cv::approxPolyDP(cv::Mat(*contour), approx, 50.0, true);
			// ãﬂéóÇ™4ê¸Ç©Ç¬ñ êœÇ™àÍíËà»è„Ç»ÇÁéläpå`
			double area = cv::contourArea(approx);
			if (approx.size() == 4 && area > 100.0) {
				cv::Mat dst(rows, cols, CV_64FC4);
				cv::Point2f dstPoints[4];

				float da = std::min(cv::norm(approx[0] - approx[1]), cv::norm(approx[2] - approx[3]));
				float db = std::min(cv::norm(approx[1] - approx[2]), cv::norm(approx[3] - approx[0]));

				if (da < db) {
					approx.push_back(approx[0]);
					approx.erase(approx.begin());
					std::cout << "objectPoints" << std::endl;
				}

				dstPoints[0] = { 0,				0 };
				dstPoints[1] = { 0,				(float)rows };
				dstPoints[2] = { (float)cols,	(float)rows };
				dstPoints[3] = { (float)cols,	0 };
				cv::Point2f aprPoints[4]{ approx[0], approx[1], approx[2], approx[3] };
				cv::polylines(frame, approx, true, cv::Scalar(255, 0, 0), 2);

				cv::putText(frame, "0", approx[0], cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 128, 0));
				cv::putText(frame, "1", approx[1], cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 128, 0));
				cv::putText(frame, "2", approx[2], cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 128, 0));
				cv::putText(frame, "3", approx[3], cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 128, 0));

				cv::Mat H = getPerspectiveTransform(aprPoints, dstPoints);
				warpPerspective(frame, dst, H, dst.size());

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

				cv::circle(frame, cube2D[0], 3, cv::Scalar(0, 0, 255));

/*
				std::cout << "objectPoints: " << objectPoints << std::endl;
				std::cout << "approx: " << approx << std::endl;
				std::cout << "cameraMatrix: " << cameraMatrix << std::endl;
				std::cout << "distCoeffs: " << distCoeffs << std::endl;
				std::cout << "rvec: " << rvec << std::endl;
				std::cout << "tvec: " << tvec << std::endl;
				std::cout << "H: " << H << std::endl;
				*/
				std::stringstream sst;
				sst << "area : " << area;
				cv::putText(frame, sst.str(), approx[0], cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 128, 0));
				cv::imshow("dst", dst);
			}
		}
		cv::imshow("frame", frame);
	}
}

