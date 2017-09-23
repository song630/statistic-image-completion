#ifndef _PATCHMATCH_H_
#define _PATCHMATCH_H_

#pragma once

#include "patch.h"
#include "segmentation.h"
#include <vector>
#include <string>

#define SIDE_LEN 3

using namespace std;
using namespace cv;

class PatchMatch {
public:
	PatchMatch(const Mat& img, const RECT& hole);
	~PatchMatch();
	void init();
	Mat propagation_search();
	void print() {
		for (int i = 0; i <= height - SIDE_LEN; i += 10)
		{
			for (int j = 0; j <= width - SIDE_LEN; j += 10)
			{
				std::cout << "[" << cur_PATCHES[i][j].get_offset().x << ", ";
				std::cout << cur_PATCHES[i][j].get_offset().y << "]" << endl;
			}
		}
	}
	typedef vector<vector<patch> > PthOfImg;
	// "cur_PATCHES": storing the most similar patch in the query image
	PthOfImg cur_PATCHES;

private:
	Mat query;
	int width, height;
	const RECT roi;  // top-left and bottom-right points marking a rectangle

	static float alpha;  // decay ratio
	static int threshold;  // the similar patch should not be in a nearby region

	typedef vector<vector<Mat> > PatchInQ;
	// "q_patches": all patches in "query", including patches within the hole
	PatchInQ q_patches;

	float get_sim(const Mat& a, const Mat& b);  // calculate similarity
	inline bool is_hole_intersect(const Point& location);
	inline bool is_within_threshold(const Point& offset);
	inline bool out_of_range(const Point& location);
};

#endif
