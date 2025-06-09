
#include <string>
#include <iostream>

#include <opencv2\opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <math.h>   

using namespace cv;
using namespace std;


typedef unsigned char byte;
typedef unsigned char byte;
typedef std::vector<std::string>::const_iterator vec_iter;
byte *matToBytes(cv::Mat image);

void odd(int x);
void even(int x);

//wrap func();
struct RGB {
	uchar blue;
	uchar green;
	uchar red;
};

#define FEATURESIZE 20;
#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
typedef struct{
	int array[20];
}wrap;


wrap func(void);

bool plotSupportVectors = true;
int numTrainingPoints = 2000;
int numTestPoints =2000;
int size = 200;
int eq = 0;
float evaluate(cv::Mat& predicted, cv::Mat& actual);
void plot_binary(cv::Mat& data, cv::Mat& classes, string name);
int f(float x, float y, int equation);
cv::Mat labelData(cv::Mat points, int equation);
void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses);
void mlp2(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses);
void writeMatToFile(cv::Mat& m, const char* filename);



int main()
{

	

	// Load a classifier from its XML description
	cv::CascadeClassifier classifier("orangeDetector11.xml");

	// Prepare a display window
	const char* const window_name{ "Oranges Detection and  Recognition Window" };

	/*cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);*/

	// Read the image file
	Mat Orangeimage = imread("image0000812.jpg", CV_LOAD_IMAGE_COLOR);


	if (Orangeimage.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}


	//Define names of the windows
	String windowORange = "Lotus";
	// Create windows with above names
	namedWindow(windowORange);
	
	// Show our images inside the created windows.
	imshow(windowORange, Orangeimage);
	
	waitKey(0); // Wait for any keystroke in the window

	destroyAllWindows(); //destroy all opened windows

	// Prepare an image where to store the video frames, and an image to store a
	// grayscale version
	// Prepare an image where to store the video frames, and an image to store a
	// grayscale version
	cv::Mat image;
	cv::Mat grayscale_image;

	// Prepare a vector where the detected features will be stored
	std::vector<cv::Rect> features;

	cv::cvtColor(image, grayscale_image, CV_BGR2GRAY);
//	cv::equalizeHist(grayscale_image, grayscale_image);

	// Detect the features in the normalized, gray-scale version of the
	// image. You don't need to clear the previously-found features because the
	// detectMultiScale method will clear before adding new features.
	//classifier.detectMultiScale(grayscale_image, features, 1.1, 2,
		//0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));


	classifier.detectMultiScale(Orangeimage, features, 1.1, 2,
		0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

	// Draw each feature as a separate green rectangle
	for (auto&& feature : features) {
		cv::rectangle(image, feature, cv::Scalar(0, 255, 0), 2);
	}

	// Show the captured image and the detected features
	cv::imshow(window_name, image);
	








	return 0;
}

//Mat GetRandomSample(Mat x)
//{
//
//
//
//}

//void writeCSV(string filename, Mat m)
//{
//	ofstream myfile;
//	myfile.open(filename.c_str());
//	myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
//	myfile.close();
//}



void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}





void mat_to_vector(Mat in, vector<float> &out){

	for (int i = 0; i < in.rows; i++) {
		for (int j = 0; j < in.cols; j++){
			//unsigned char temp;

			//file << Dst.at<float>(i,j)  << endl;
			out.push_back(in.at<float>(i, j));
		}
	}

}
void vector_to_mat(vector<float> in, Mat out, int cols, int rows){
	for (int i = rows - 1; i >= 0; i--) {
		for (int j = cols - 1; j >= 0; j--){

			out.at<float>(i, j) = in.back();
			in.pop_back();
			//file << Dst.at<float>(i,j)  << endl;
			// dst_temp.push_back(Dst.at<float>(i,j));
		}
	}
}

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i, 0);
		double a = actual.at<double>(i, 0);
		if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		}
		else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}

// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
	cv::Mat plot(size, size, CV_8UC3);
	plot.setTo(cv::Scalar(255.0, 255.0, 255.0));
	for (int i = 0; i < data.rows; i++) {

		float x = data.at<float>(i, 0) * size;
		float y = data.at<float>(i, 1) * size;

		if (classes.at<float>(i, 0) > 0) {
			cv::circle(plot, Point(x, y), 2, CV_RGB(255, 0, 0), 1);
		}
		else {
			cv::circle(plot, Point(x, y), 2, CV_RGB(0, 255, 0), 1);
		}
	}
	cv::imshow(name, plot);
}

// function to learn
int f(float x, float y, int equation) {
	switch (equation) {
	case 0:
		return y > sin(x * 10) ? -1 : 1;
		break;
	case 1:
		return y > cos(x * 10) ? -1 : 1;
		break;
	case 2:
		return y > 2 * x ? -1 : 1;
		break;
	case 3:
		return y > tan(x * 10) ? -1 : 1;
		break;
	default:
		return y > cos(x * 10) ? -1 : 1;
	}
}

// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
	cv::Mat labels(points.rows, 1, CV_32FC1);
	for (int i = 0; i < points.rows; i++) {
		float x = points.at<float>(i, 0);
		float y = points.at<float>(i, 1);
		labels.at<float>(i, 0) = f(x, y, equation);
	}
	return labels;
}


void mlp2(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {


	/////this is a four layer neural network
	cv::Mat layers = cv::Mat(3, 1, CV_32SC1);

	//int koko=layers.rowRange;
	/*cout << "Rows Range in Layers" << koko << endl;*/

	//layers.row(0) = cv::Scalar(2);  //input layer accepts x and y // two because the Training Data is only two columns or two features////
	layers.row(0) = cv::Scalar(4);  //input layer accepts x and y // two because the Training Data is only two columns or two features

	layers.row(1) = cv::Scalar(21);  //hidden layer
	layers.row(2) = cv::Scalar(11);  //hidden layer  //output layer
	//layers.row(3) = cv::Scalar(1);    //output layer returns 1 or -1  // The number of classes

	CvANN_MLP mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 500;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	int activateFunc = CvANN_MLP::SIGMOID_SYM;

	//mlp.create(layers);
	mlp.create(layers, activateFunc, 0, 0);

	// train
	int numLayers = mlp.get_layer_count();
	cout << "number of neural network Layers  Are   " << numLayers << "  Layers" << endl;

	cout << "Cooling Down the neural network is being trained" << endl;
	mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);

	cout << "Training is Finished" << endl;
	cv::Mat results(3960, 11, CV_32F);

	mlp.predict(testData, results);


	//int accuracy=evaluate(results, testClasses);


	//cout << "Accuracy is equal to " << accuracy << endl;


	const CvMat* layersizes101 = mlp.get_layer_sizes();

	//cv::Mat response(1, 11, CV_32FC1);
	//cv::Mat predicted(testClasses.rows, 1, CV_32F);
	//for (int i = 0; i < testData.rows; i++) {
	//	cv::Mat response(1, 1, CV_32FC1);
	//	cv::Mat sample = testData.row(i);

	//	mlp.predict(sample, response);

	//	/*	cout << " Output is " << response << endl;

	//	cout << "Size of REsponse " << response.rows << response.cols   <<endl;

	//	cout << "Out Is " << response.at<double>(0, 0) << endl;*/

	//	//		int koko = (int)response.at<float>(0, 0);

	//	//	cout << "Out Is " <<koko << endl;

	//	//// We need to loop here over the neurons of the output
	//	//////////////////////////////////////////////////////////////

	//	predicted.at<float>(i, 0) = response.at<double>(0, 0);

	//}


	//for (int i = 0; i < 20; i++)
	//{

	//	cout << "result output " << results.at<double>(i, 0) << endl;

	//}

	double* weights0 = mlp.get_weights(0);
	cout << "Weights in layer 0" << *(weights0) << endl;
	cout << "Weights  Weights" << endl;
	/*cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
*/

	cout << "Prediction done!" << endl;
	cout << endl << "Classification result: " << endl << results << endl;


	cv::FileStorage file("ANN_Results.txt", cv::FileStorage::WRITE);
	cv::Mat someMatrixOfAnyType;

	cv::Mat B(3960, 1, CV_32F);


	results.convertTo(B, CV_64F, 0.01); // Convert back to double with a scaling of 0.0001

	results.convertTo(B, CV_8U);
	// Write to file!
	file << "matName" << B;


	cout << B << endl;

	//	//We need to know where in output is the max val, the x (cols) is the class.
	Point maxLoc;
	double maxVal;
	minMaxLoc(results, 0, &maxVal, 0, &maxLoc);

	cout << "Accuracy Is " << maxLoc.x << maxVal << endl;

	/*plot_binary(testData, predicted, "Predictions Backpropagation");*/


	/*return results;*/
}



void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {


	/////this is a four layer neural network
	cv::Mat layers = cv::Mat(3, 1, CV_32SC1);

	//layers.row(0) = cv::Scalar(2);  //input layer accepts x and y // two because the Training Data is only two columns or two features////
	layers.row(0) = cv::Scalar(4);  //input layer accepts x and y // two because the Training Data is only two columns or two features
	
	layers.row(1) = cv::Scalar(21);  //hidden layer
	layers.row(2) = cv::Scalar(1);  //hidden layer  //output layer
	//layers.row(3) = cv::Scalar(1);    //output layer returns 1 or -1  // The number of classes

	CvANN_MLP mlp;
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 500;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;
		
	int activateFunc = CvANN_MLP::SIGMOID_SYM;

	//ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	//ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));
	//ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);

	/*int layerSize=mlp.get_layer_sizes;*/



	//mlp.create(layers);
     mlp.create(layers, activateFunc, 0, 0);

	// train
	 int numLayers = mlp.get_layer_count();
	 cout << "number of neural network Layers  Are   " << numLayers << "  Layers" << endl;

	 cout << "Cool Downing the neural network is being trained" << endl;
	mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);

	cout << "Training is Finished" << endl;
	cv::Mat results(3960, 1, CV_32F);
	
	mlp.predict(testData, results);


	


	const CvMat* layersizes101 = mlp.get_layer_sizes();

	cv::Mat response(1, 1, CV_32FC1);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		cv::Mat response(1, 1, CV_32FC1);
		cv::Mat sample = testData.row(i);

		mlp.predict(sample, response);

	/*	cout << " Output is " << response << endl;

		cout << "Size of REsponse " << response.rows << response.cols   <<endl;
		
		cout << "Out Is " << response.at<double>(0, 0) << endl;*/

//		int koko = (int)response.at<float>(0, 0);
	
	//	cout << "Out Is " <<koko << endl;

		//// We need to loop here over the neurons of the output
		//////////////////////////////////////////////////////////////

		predicted.at<float>(i, 0) = response.at<double>(0, 0);

	}


	//for (int i = 0; i < 20; i++)
	//{

	//	cout << "result output " << results.at<double>(i, 0) << endl;

	//}

	double* weights0=mlp.get_weights(0);
	cout << "Weights in layer 0" << *(weights0) << endl;
	cout << "Weights  Weights" << endl;
	cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;


	cout << "Prediction done!" << endl;
	cout << endl << "Classification result: " << endl << results << endl;


	cv::FileStorage file("ANN_Results.txt", cv::FileStorage::WRITE);
	cv::Mat someMatrixOfAnyType;

	cv::Mat B(3960, 1, CV_32F);


	results.convertTo(B, CV_64F, 0.01); // Convert back to double with a scaling of 0.0001

	results.convertTo(B, CV_8U);
	// Write to file!
	file << "matName" << B;


	cout << B << endl;

	//	//We need to know where in output is the max val, the x (cols) is the class.
	Point maxLoc;
	double maxVal;
	minMaxLoc(results, 0, &maxVal, 0, &maxLoc);

	cout << "Accuracy Is " << maxLoc.x <<  maxVal << endl;

	/*plot_binary(testData, predicted, "Predictions Backpropagation");*/
}

void mse()
{

	int sum_sq = 0;
	double mse;

	/*for (i = 0; i < h; ++i)
	{
		for (j = 0; j < w; ++j)
		{
			int p1 = image1[i][j];
			int p2 = image2[i][j];
			int err = p2 - p1;
			sum_sq += (err * err);
		}
	}
	mse = (double)sum_sq / (h * w);
*/
}

void resizeCol(Mat& m, size_t sz, const Scalar& s)
{
	Mat tm(m.rows, m.cols + sz, m.type());
	tm.setTo(s);
	m.copyTo(tm(Rect(Point(0, 0), m.size())));
	m = tm;
}


int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname) break;
		++index;
	}
	return index;
}








//std::vector<byte> matToBytes(cv::Mat image)
//{
//	int size = image.total() * image.elemSize();
//	std::vector<byte> img_bytes(size);
//	img_bytes.assign(image.datastart, image.dataend);
//	return img_bytes;
//}

cv::Mat bytesToMat(vector<byte> bytes, int width, int height)
{
	cv::Mat image(height, width, CV_8UC3, bytes.data());
	return image;
}




byte * matToBytes(Mat image)
{
	int size = image.total() * image.elemSize();
	byte * bytes = new byte[size];  //delete[] later
	std::memcpy(bytes, image.data, size * sizeof(byte));
	return bytes;
}







Mat bytesToMat(byte * bytes, int width, int height)
{
	Mat image = Mat(height, width, CV_8UC3, bytes).clone(); // make a copy
	return image;
}




/**
* Get a binary code associated to a class
*/
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = getClassId(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

/**
* Receives a column matrix contained the probabilities associated to
* each class and returns the id of column which contains the highest
* probability
*/
int getPredictedClass(const cv::Mat& predictions)
{
	float maxPrediction = predictions.at<float>(0);
	float maxPredictionIndex = 0;
	const float* ptrPredictions = predictions.ptr<float>(0);
	for (int i = 0; i < predictions.cols; i++)
	{
		float prediction = *ptrPredictions++;
		if (prediction > maxPrediction)
		{
			maxPrediction = prediction;
			maxPredictionIndex = i;
		}
	}
	return maxPredictionIndex;
}

/**
* Get a confusion matrix from a set of test samples and their expected
* outputs
*/
//std::vector<std::vector<int> > getConfusionMatrix(cv::Ptr<cv::ml::ANN_MLP> mlp,
//	const cv::Mat& testSamples, const std::vector<int>& testOutputExpected)
//{
//	cv::Mat testOutput;
//	mlp->predict(testSamples, testOutput);
//	std::vector<std::vector<int> > confusionMatrix(2, std::vector<int>(2));
//	for (int i = 0; i < testOutput.rows; i++)
//	{
//		int predictedClass = getPredictedClass(testOutput.row(i));
//		int expectedClass = testOutputExpected.at(i);
//		confusionMatrix[expectedClass][predictedClass]++;
//	}
//	return confusionMatrix;
//}

/**
* Print a confusion matrix on screen
*/
void printConfusionMatrix(const std::vector<std::vector<int> >& confusionMatrix,
	const std::set<std::string>& classes)
{
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		std::cout << *it << " ";
	}
	std::cout << std::endl;
	for (size_t i = 0; i < confusionMatrix.size(); i++)
	{
		for (size_t j = 0; j < confusionMatrix[i].size(); j++)
		{
			std::cout << confusionMatrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

/**
* Get the accuracy for a model (i.e., percentage of correctly predicted
* test samples)
*/
float getAccuracy(const std::vector<std::vector<int> >& confusionMatrix)
{
	int hits = 0;
	int total = 0;
	for (size_t i = 0; i < confusionMatrix.size(); i++)
	{
		for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
		{
			if (i == j) hits += confusionMatrix.at(i).at(j);
			total += confusionMatrix.at(i).at(j);
		}
	}
	return hits / (float)total;
}











