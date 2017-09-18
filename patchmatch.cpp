#include "patch.h"
#include "patchmatch.h"
#include <ctime>

#define IMG_PATH "D://from_ImageNet/"  // the file where the images are saved
#define ITERATIONS 9
#define MAX_SIMIARITY 1e6
#define MAX2(a, b) (((a) > (b)) ? (a) : (b))
#define MIN2(a, b) (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

float PatchMatch::alpha = 0.5;
int PatchMatch::threshold = 8;

// ===== threshold ?

PatchMatch::PatchMatch(const Mat& img, const RECT& hole) : query(img), roi(hole)  // ctor
{
	width = query.cols;
	height = query.rows;
	if (width <= SIDE_LEN || height <= SIDE_LEN)
	{  // the image is smaller than a patch
		cout << "Incorrect image size: " << endl;
		cout << "width: " << width << ", height: " << height << endl;
		system("pause");
		exit(1);
	}
	// notice: in "Point", x-column, y-row.

	q_patches.resize(height - SIDE_LEN + 1);  // resize "q_patches"
	cur_PATCHES.resize(height - SIDE_LEN + 1);  // resize "cur_PATCHES"
	for (int i = 0; i <= height - SIDE_LEN; i++)
	{
		q_patches[i].resize(width - SIDE_LEN + 1);
		cur_PATCHES[i].resize(width - SIDE_LEN + 1);
		for (int j = 0; j <= width - SIDE_LEN; j++)
		{
			q_patches[i][j] = query(Rect(j, i, SIDE_LEN, SIDE_LEN));  // get all patches in the image
			cur_PATCHES[i].push_back(patch(Point(j, i)));  // call ctor
		}
	}
}

PatchMatch::~PatchMatch()
{
	PthOfImg::iterator i1;
	for (i1 = cur_PATCHES.begin(); i1 != cur_PATCHES.end(); i1++)
		(*i1).clear();
	cur_PATCHES.clear();

	PatchInQ::iterator i3;
	for (i3 = q_patches.begin(); i3 != q_patches.end(); i3++)
		(*i3).clear();
	q_patches.clear();
}

// set "location" and "offset", and compute "similarity"
void PatchMatch::init()
{
	// randomly assign each patch (patches intersecting with the hole not included) a relative one
	int row_end = height - SIDE_LEN;
	int col_end = width - SIDE_LEN;
	for (int i = 0; i <= row_end; i++)
	{
		for (int j = 0; j <= col_end; j++)
		{
			if (!is_hole_intersect(Point(j, i)))  // not intersect with the hole
			{
				// rand_x, rand_y: coordinates of the randomly assigned patch
				int rand_x = std::rand() % (width - SIDE_LEN);  // 0 to width - 3
				int rand_y = std::rand() % (height - SIDE_LEN);  // 0 to height - 3
				while (is_hole_intersect(Point(rand_x, rand_y)) ||
					is_within_threshold(Point(rand_x - j, rand_y - i)))
				{  // until finding a legitimate patch
					rand_x = std::rand() % (width - SIDE_LEN);
					rand_y = std::rand() % (height - SIDE_LEN);
				}
				// then update:
				cur_PATCHES[i][j].update(Point(rand_x - j, rand_y - i),
					get_sim(q_patches[i][j], q_patches[rand_y][rand_x]));
			}  // end-if
		}  // end-inner loop
	}
}

Mat PatchMatch::propagation_search()
{
	for (int i = 1; i <= ITERATIONS; i++)
	{
		cout << "begin iteration " << i << endl;
		if (i % 2 == 1)  // odd iterations
		{
			PthOfImg::iterator iter1 = cur_PATCHES.begin();
			for (int j = 0; iter1 != cur_PATCHES.end(); iter1++, j++)
			{  // traverse every patch
				vector<patch>::iterator iter2 = (*iter1).begin();
				for (int k = 0; iter2 != (*iter1).end(); iter2++, k++)
				{
					if (k == 0 && j == 0)  // the top-left corner
						continue;
					int dst_x1 = -1, dst_y1 = -1, dst_x2 = -1, dst_y2 = -1;
					if (k != 0 && j != 0)  // not the first row or col
					{
						// "dst_x1": the relative patch of the patch 1 pixel left of current patch
						dst_x1 = cur_PATCHES[j][k - 1].get_offset().x + k;
						dst_y1 = cur_PATCHES[j][k - 1].get_offset().y + j;
						// "dst_x2": the relative patch of the patch 1 pixel above current patch
						dst_x2 = cur_PATCHES[j - 1][k].get_offset().x + k;
						dst_y2 = cur_PATCHES[j - 1][k].get_offset().y + j;
					}
					else if (j == 0 && k != 0)  // the first row
					{
						// "dst_x1": the relative patch of the patch 1 pixel left of current patch
						dst_x1 = cur_PATCHES[j][k - 1].get_offset().x + k;
						dst_y1 = cur_PATCHES[j][k - 1].get_offset().y + j;
					}
					else if (j != 0 && k == 0)  // the first col
					{
						// "dst_x2": the relative patch of the patch 1 pixel above current patch
						dst_x2 = cur_PATCHES[j - 1][k].get_offset().x + k;
						dst_y2 = cur_PATCHES[j - 1][k].get_offset().y + j;
					}
					float s1, s2;  // difference of patch1 and patch2
					if (is_within_threshold(Point(dst_x1, dst_y1)) ||
						is_hole_intersect(Point(dst_x1, dst_y1)) ||
						out_of_range(Point(dst_x1, dst_y1)))
						s1 = MAX_SIMIARITY;
					else
						s1 = get_sim(q_patches[j][k], q_patches[dst_y1][dst_x1]);
					if (is_within_threshold(Point(dst_x2, dst_y2)) ||
						is_hole_intersect(Point(dst_x2, dst_y2)) ||
						out_of_range(Point(dst_x2, dst_y2)))
						s2 = MAX_SIMIARITY;
					else
						s2 = get_sim(q_patches[j][k], q_patches[dst_y2][dst_x2]);

					// then compare and update the mapping
					float min_s = std::min((*iter2).get_sim(), MIN2(s1, s2));
					if (min_s == s1)  // left
						(*iter2).update(Point(dst_x1 - k, dst_y1 - j), s1);
					else if (min_s == s2)  // up
						(*iter2).update(Point(dst_x2 - k, dst_y2 - j), s2);
				}
			}
		}
		else  // even iterations, scan reversely
		{
			for (int j = height - SIDE_LEN; j >= 0; j--)  // scan from the last row
			{
				for (int k = width - SIDE_LEN; k >= 0; k--)  // scan from the last column
				{
					if (j == height - SIDE_LEN && k == width - SIDE_LEN)  // the bottom-right corner
						continue;
					int dst_x1 = -10, dst_y1 = -10, dst_x2 = -10, dst_y2 = -10;
					if (j != height - SIDE_LEN && k != width - SIDE_LEN)  // not the last row or col
					{
						// "dst_x1": the relative patch of the patch 1 pixel right of current patch
						dst_x1 = cur_PATCHES[j][k + 1].get_offset().x + k;
						dst_y1 = cur_PATCHES[j][k + 1].get_offset().y + j;
						// "dst_x2": the relative patch of the patch 1 pixel below current patch
						dst_x2 = cur_PATCHES[j + 1][k].get_offset().x + k;
						dst_y2 = cur_PATCHES[j + 1][k].get_offset().y + j;
					}
					else if (j == height - SIDE_LEN && k != width - SIDE_LEN)  // the last row
					{
						// "dst_x1": the relative patch of the patch 1 pixel right of current patch
						dst_x1 = cur_PATCHES[j][k + 1].get_offset().x + k;
						dst_y1 = cur_PATCHES[j][k + 1].get_offset().y + j;
					}
					else if (j != height - SIDE_LEN && k == width - SIDE_LEN)  // the last col
					{
						// "dst_x2": the relative patch of the patch 1 pixel below current patch
						dst_x2 = cur_PATCHES[j + 1][k].get_offset().x + k;
						dst_y2 = cur_PATCHES[j + 1][k].get_offset().y + j;
					}
					float s1, s2;
					if (is_within_threshold(Point(dst_x1, dst_y1)) ||
						is_hole_intersect(Point(dst_x1, dst_y1)) ||
						out_of_range(Point(dst_x1, dst_y1)))
						s1 = MAX_SIMIARITY;
					else
						s1 = get_sim(q_patches[j][k], q_patches[dst_y1][dst_x1]);
					if (is_within_threshold(Point(dst_x2, dst_y2)) ||
						is_hole_intersect(Point(dst_x2, dst_y2)) ||
						out_of_range(Point(dst_x2, dst_y2)))
						s2 = MAX_SIMIARITY;
					else
						s2 = get_sim(q_patches[j][k], q_patches[dst_y2][dst_x2]);

					// then compare and update the mapping
					float min_s = std::min(cur_PATCHES[j][k].get_sim(), MIN2(s1, s2));
					if (min_s == s1)  // right
						cur_PATCHES[j][k].update(Point(dst_x1 - k, dst_y1 - j), s1);
					else if (min_s == s2)  // down
						cur_PATCHES[j][k].update(Point(dst_x2 - k, dst_y2 - j), s2);
				}
			}
		}
		cout << "random search in iteration " << i << endl;
		// random search:
		int rand_x, rand_y;  // coordinates of patch2
		float s;  // similarity
		PthOfImg::iterator iter1 = cur_PATCHES.begin();
		for (int j = 0; iter1 != cur_PATCHES.end(); iter1++, j++)
		{
			vector<patch>::iterator iter2 = (*iter1).begin();
			for (int k = 0; iter2 != (*iter1).end(); iter2++, k++)
			{
				int search_x = width - SIDE_LEN + 1;  // init search radius
				int search_y = height - SIDE_LEN + 1;
				search_x >>= 1;
				search_y >>= 1;
				int left_bound, right_bound, up_bound, down_bound;
				while (search_x > 3 && search_y > 3)  // break when radius is less than 1 pixel
				{  // first compute the range:
					Point off = (*iter2).get_offset();
					left_bound = MAX2(0, k + off.x - search_x);
					right_bound = MIN2(width - SIDE_LEN, k + off.x + search_x);
					up_bound = MAX2(0, j + off.y - search_y);
					down_bound = MIN2(height - SIDE_LEN, j + off.y + search_y);
					srand((unsigned)time(NULL));  // then get a random patch:
					rand_y = std::rand() % (down_bound - up_bound + 1) + up_bound;
					rand_x = std::rand() % (right_bound - left_bound + 1) + left_bound;
					while (is_hole_intersect(Point(rand_x, rand_y)) ||
						is_within_threshold(Point(rand_x - k, rand_y - j)))
					{
						rand_y = std::rand() % (down_bound - up_bound + 1) + up_bound;
						rand_x = std::rand() % (right_bound - left_bound + 1) + left_bound;
					}
					s = get_sim(q_patches[j][k], q_patches[rand_y][rand_x]);
					if (s < (*iter2).get_sim())  // at last update
						(*iter2).update(Point(rand_x - k, rand_y - j), s);
					search_x >>= 1;
					search_y >>= 1;
				}
			}
		}
	}  // end ITERATIONS
	return query;
}

bool PatchMatch::is_hole_intersect(const Point& location)
{
	if (location.x + SIDE_LEN - 1 >= roi.first.x && location.x <= roi.second.x &&
		location.y + SIDE_LEN - 1 >= roi.first.y && location.y <= roi.second.y)
		return true;
	else
		return false;
}

bool PatchMatch::is_within_threshold(const Point& offset)
{
	if (offset.x * offset.x + offset.y * offset.y <= threshold * threshold)
		return true;
	else
		return false;
}

bool PatchMatch::out_of_range(const Point& location)
{
	if (location.x < 0 || location.x > width - SIDE_LEN ||
		location.y < 0 || location.y > height - SIDE_LEN)
		return true;
	else
		return false;
}

float PatchMatch::get_sim(const Mat& a, const Mat& b)  // private
{
	Mat rst;  // an 1 * 4 Scaler
	absdiff(a, b, rst);  // compute the difference
	return static_cast<float>(sum(sum(rst))[0]);
}
