#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include "patchmatch.h"
#include "kernel.h"
#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

typedef vector<Point> POINTS;

class histogram {
public:
	histogram(vector<vector<patch> >& cur_PATCHES, const Mat& src) : STATS(cur_PATCHES),
		img_height(src.rows), img_width(src.cols) {
		// CV_16SC2: signed, 2-channel, -32768->32767
		mat_of_offsets = Mat(img_height - SIDE_LEN + 1, img_width - SIDE_LEN + 1, 
			CV_16SC2, Scalar::all(0));  // (rows, cols)
		// then assign values to the Mat:
		for (int i = 0; i <= mat_of_offsets.rows - 1; i++)
		{
			Vec2s *p = mat_of_offsets.ptr<Vec2s>(i);
			for (int j = 0; j <= mat_of_offsets.cols - 1; j++)
			{  // Vec2s is (short, short)
				p[j][0] = static_cast<short>(STATS[i][j].get_offset().x);  // col
				p[j][1] = static_cast<short>(STATS[i][j].get_offset().y);  // row
			}
		}
	}

	~histogram() {
		vector<vector<patch> >::iterator i;
		for (i = STATS.begin(); i != STATS.end(); i++)
			(*i).clear();
		STATS.clear();
	}

	void computeHist() {
		int channels[] = { 0, 1 };
		int bins[] = { img_width / SIDE_LEN, img_height / SIDE_LEN };  // groups  // order?
		float row_ranges[] = { -(img_height - SIDE_LEN), img_height - SIDE_LEN };  // range of x coordinate
		float col_ranges[] = { -(img_width - SIDE_LEN), img_width - SIDE_LEN };  // range of y coordinate
		const float *Ranges[] = { col_ranges, row_ranges };  // order?
		calcHist(&mat_of_offsets, 1, channels, Mat(), dstHist, 2, bins, Ranges, true, false);
	}

	void drawHist() {
		hist = Mat::zeros(Size((img_width - SIDE_LEN) * 2 + 1, (img_height - SIDE_LEN) * 2 + 1), 
			CV_8UC1);  // Size(width, height). a single-channel, gray-scale img.
		// ===== caution: the image "hist" and normalization are only used for illustration,
		// when dealing with the statistics, just use "dstHist".
		double g_dHistMaxValue;
		minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
		for (int i = 0; i < dstHist.rows; i++)
		{
			uchar *p = hist.ptr<uchar>(i);
			for (int j = 0; j < dstHist.cols; j++)
			{
				int value = cvRound(dstHist.at<float>(Point(j, i)) * 256 / g_dHistMaxValue);
				p[j] = value;
			}
		}
		namedWindow("histogram");
		imshow("histogram", hist);
		waitKey(0);
	}

	void convoluteHist(const Kernel& K) {  // compute convolution

	}

	void getPeaks(const int num_peaks) {

	}

	POINTS peaks;  // get the dominant offsets from the statistics

private:
	vector<vector<patch> > STATS;  // all the raw statistics
	const int img_height;  // decide the num of bins
	const int img_width;
	Mat mat_of_offsets;  // a 2-channel Mat, constructed from STATS[]
	MatND dstHist;  // store the result
	Mat hist;  // draw
};

#endif
