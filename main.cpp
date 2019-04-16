#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;

int main() {

	const int cols = 256;
	const int rows = (int)(cols * 89.0/58.0);

	//camera parameters
	double fx = 525.0; //focal length x
	double fy = 525.0; //focal le

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

	cv::VideoCapture cap(0);
	// esc ‚ğ‰Ÿ‚·‚Ü‚Å
	while (cv::waitKey(5) != 0x1b) {
		cv::Mat frame;
		cap >> frame;
		// ƒLƒƒƒvƒ`ƒƒ‚Å‚«‚Ä‚¢‚È‚¯‚ê‚Îˆ—‚ğ”ò‚Î‚·
		if (!frame.data) {
			continue;
		}

		// 2’l‰»
		cv::Mat grayImage, binImage;
		cv::cvtColor(frame, grayImage, COLOR_BGR2GRAY);
		cv::threshold(grayImage, binImage, 128.0, 255.0, THRESH_OTSU);
		cv::imshow("bin", binImage);

		// —ÖŠs’Šo
		std::vector< std::vector< cv::Point > > contours;
		cv::findContours(binImage, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		// ŒŸo‚³‚ê‚½—ÖŠsü‚Ì•`‰æ
		for (auto contour = contours.begin(); contour != contours.end(); contour++) {
			cv::polylines(frame, *contour, true, cv::Scalar(0, 255, 0), 2);
		}

		// —ÖŠs‚ªlŠpŒ`‚©‚Ì”»’è
		for (auto contour = contours.begin(); contour != contours.end(); contour++) {
			// —ÖŠs‚ğ’¼ü‹ß—
			std::vector< cv::Point > approx;
			cv::approxPolyDP(cv::Mat(*contour), approx, 50.0, true);
			// ‹ß—‚ª4ü‚©‚Â–ÊÏ‚ªˆê’èˆÈã‚È‚çlŠpŒ`
			double area = cv::contourArea(approx);
			if (approx.size() == 4 && area > 100.0) {
				cv::Mat dst(rows, cols, CV_64FC4);
				cv::Point2f dstPoints[4];

				float da = std::min(cv::norm(approx[0] - approx[1]), cv::norm(approx[2] - approx[3]));
				float db = std::min(cv::norm(approx[1] - approx[2]), cv::norm(approx[3] - approx[0]));	

				if (da > db) {
					dstPoints[0] = { 0,				0 };
					dstPoints[1] = { 0,				(float)rows };
					dstPoints[2] = { (float)cols,	(float)rows };
					dstPoints[3] = { (float)cols,	0 };
				} else {
					dstPoints[0] = { (float)cols,	0 };
					dstPoints[1] = { 0,				0 };
					dstPoints[2] = { 0,				(float)rows };
					dstPoints[3] = { (float)cols,	(float)rows };
				}
				cv::Point2f aprPoints[4]{ approx[0], approx[1], approx[2], approx[3] };
				cv::polylines(frame, approx, true, cv::Scalar(255, 0, 0), 2);

				std::stringstream dist1;
				dist1 << da;
				std::stringstream dist2;
				dist2 << db;
				cv::putText(frame, "0", approx[0], FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 128, 0));
				cv::putText(frame, dist1.str(), approx[1], FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 128, 0));
				cv::putText(frame, "2", approx[2], FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 128, 0));
				cv::putText(frame, dist2.str(), approx[3], FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 128, 0));

				cv::Mat H = getPerspectiveTransform(aprPoints, dstPoints);
				warpPerspective(frame, dst, H, dst.size());

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

				std::cout << "objectPoints: " << objectPoints << std::endl;
				std::cout << "approx: " << approx << std::endl;
				std::cout << "cameraMatrix: " << cameraMatrix << std::endl;
				std::cout << "distCoeffs: " << distCoeffs << std::endl;
				cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

				std::cout << "rvec: " << rvec << std::endl;
				std::cout << "tvec: " << tvec << std::endl;
				std::cout << "H: " << H << std::endl;


				std::stringstream sst;
				sst << "area : " << area;
				cv::putText(frame, sst.str(), approx[0], FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 128, 0));
				cv::imshow("dst", dst);
			}
		}
		cv::imshow("frame", frame);
	}


}