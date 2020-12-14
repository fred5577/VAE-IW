#include "screen.h"
#include <iostream>
#include <string>
#include <cassert>
// Copyright (c) 2019, Blai Bonet
// All rights reserved.
// Code from https://github.com/bonetblai/rollout-iw/blob/master/src/screen.h

int width = 160;
int height = 210;
std::vector<bool> basic_features_bitmap_;
std::vector<bool> bprot_features_bitmap_;

static const size_t width_ = 160;
static const size_t height_ = 210;
static const size_t num_basic_features_ = 16 * 14 * 128; // 28,672
static const size_t num_bpros_features_t0_ = 6803136; // (dc,dr,k1,k2) where k1 < k2, number equal to 31 * 27 * 128 * 127 / 2
static const size_t num_bpros_features_t1_ = 53504; // (dc,dr,k,k) where dc != 0 or dr != 0, number equal to (31 * 27 - 1) * 128 / 2
static const size_t num_bpros_features_t2_ = 128; // (dc,dr,k,k) where dc = dr = 0, number equal to 128
static const size_t num_bpros_features_ = num_bpros_features_t0_ + num_bpros_features_t1_ + num_bpros_features_t2_; // 6,856,768
static const size_t num_bprot_features_ = 31 * 27 * 128 * 128; // 13,713,408
typedef std::pair<size_t, size_t> patch_t;
typedef std::pair<patch_t, int> basic_feature_t;
typedef std::pair<size_t, size_t> offset_t;
typedef std::pair<offset_t, std::pair<int, int> > bpros_feature_t;
typedef std::pair<offset_t, std::pair<int, int> > bprot_feature_t;

std::vector<bool> bpros_features_bitmap_;


int calculate_bpros(int c1, int r1, int c2, int r2, int p1, int p2) {
	int dc = c1 - c2;
	int dr = r1 - r2;
	if (p1 > p2) {
		dc = -dc;
		dr = -dr;
		int temp = p2;
		p2 = p1;
		p1 = temp;
	}
	
	

	int offsetcol = c1 + c2 + 1;
	int offsetrow = r1 + r2 + 1;
	int pack = 0;
	if (p1 < p2 || p2 < p1) {
		//return ((offsetcol * offsetrow * (p1 + 1) * (p2 + 1) ) / 2) - 1;
		pack = ((15 + dc) * 27 + (13 + dr)) * 128 * 127 / 2;
		pack += p1 * 127 - p1 * (1 + p1) / 2 + p2 - 1;
		return pack;
	}

	if ((dc != 0) || (dr != 0)) {
		if ((dc < 0) || ((dc == 0) && (dr < 0))) {
			dc = -dc;
			dr = -dr;
		}

		if (dc > 0) {
			pack = ((dc - 1) * 27 + (13 + dr)) * 128 + p1;
		}
		else {
			pack = 15 * 27 * 128 + (dr - 1) * 128 + p1;
		}

		pack += num_bpros_features_t0_;
		return pack;
	}
	else {
		if (p1 < p2)
			return p1 + num_bpros_features_t0_ + num_bpros_features_t1_;
		else
			return p2 + num_bpros_features_t0_ + num_bpros_features_t1_;
		//return (((offsetcol * offsetrow - 1) * 128) / 2) + num_bpros_features_t0_;
	}


	//return ((offsetcol * offsetrow * (p1 + 1) * (p2 + 1) - 128) / 2) + 128;
}

int calculate_bprot(int c1, int r1, int c2, int r2, int p1, int p2) {
	int offsetcol = c1 + c2 + 1;
	int offsetrow = r1 + r2 + 1;

	if (p1 == 0 && p2 == 0) {
		return offsetcol * offsetrow;
	}
	else if (p1 == 0) {
		return offsetcol * offsetrow * p2;
	}
	else if (p2 == 0) {
		return offsetcol * offsetrow * p1;
	}
	return offsetcol * offsetrow * p1 * p2;
}

static bool is_bpros_feature(int pack) {
	return (pack >= int(num_basic_features_)) && (pack < int(num_basic_features_ + num_bpros_features_));
}

static bool is_basic_feature(int pack) {
	return (pack >= 0) && (pack < int(num_basic_features_));
}

static int pack_basic_feature(size_t c, size_t r, int p) {
	//std::cout << "c:" << c << "r:" << r << "p:" << p << "\n";
	int pack = ((14 * c + r) << 7) + p;
	assert(is_basic_feature(pack));
	return pack;
}

static void unpack_basic_feature(int pack, basic_feature_t& bf) {
	bf.first.first = (pack >> 7) / 14;
	bf.first.second = (pack >> 7) % 14;
	bf.second = pack & 127;
}
static int pack_bpros_feature(int dc, int dr, int p1, int p2) {

	int pack = 0;
	if (p1 < p2) {
		pack = ((15 + dc) * 27 + (13 + dr)) * 128 * 127 / 2;
		pack += p1 * 127 - p1 * (1 + p1) / 2 + p2 - 1;
	}
	else if ((dc != 0) || (dr != 0)) {
		if ((dc < 0) || ((dc == 0) && (dr < 0))) {
			dc = -dc;
			dr = -dr;
		}

		if (dc > 0) {
			pack = ((dc - 1) * 27 + (13 + dr)) * 128 + p1;
		}
		else {
			pack = 15 * 27 * 128 + (dr - 1) * 128 + p1;
		}

		pack += num_bpros_features_t0_;
	}
	else {
		pack = p1;
		pack += num_bpros_features_t0_ + num_bpros_features_t1_;
	}
	pack += num_basic_features_;
	return pack;
}

static int pack_bpros_feature(const basic_feature_t& bf1, const basic_feature_t& bf2) {
	int dc = bf1.first.first - bf2.first.first;
	int dr = bf1.first.second - bf2.first.second;
	if (bf1.second <= bf2.second)
		return pack_bpros_feature(dc, dr, bf1.second, bf2.second);
	else
		return pack_bpros_feature(-dc, -dr, bf2.second, bf1.second);
}

static int pack_bprot_feature(int dc, int dr, int p1, int p2) {
	assert((-15 <= dc) && (dc <= 15));
	assert((-13 <= dr) && (dr <= 13));
	assert((((15 + dc) * 27 + (13 + dr)) * 128 + p1) * 128 + p2 < int(num_bprot_features_));
	return num_basic_features_ + num_bpros_features_ + (((15 + dc) * 27 + (13 + dr)) * 128 + p1) * 128 + p2;
}

static int pack_bprot_feature(const basic_feature_t& bf1, const basic_feature_t& bf2) {
	int dc = bf1.first.first - bf2.first.first;
	int dr = bf1.first.second - bf2.first.second;
	return pack_bprot_feature(dc, dr, bf1.second, bf2.second);
}

void compute_basic_features(size_t c, size_t r, std::vector<int> screen,std::vector<int>* screen_state_atoms = 0) {

	for (size_t ic = 0; ic < 10; ++ic) {
		for (size_t ir = 0; ir < 15; ++ir) {
			int rows = (15 * r + ir);
			int cols = (10 * c + ic);
			int p = 0;
			//std::cout << "cols: " << cols << "rows:" << rows << "screen:"<< screen[rows];
			if (rows == 0) {
				p = screen[cols];
			} else if(cols == 0) {
				p = screen[rows];
			} else {
				p = screen[rows * cols];
			}//(15 * r + ir, 10 * c + ic);
			// p += 2;
			int pack = pack_basic_feature(c, r, p >> 1);
			if (!basic_features_bitmap_[pack]) {
				basic_features_bitmap_[pack] = true;
				if (screen_state_atoms != nullptr)
					screen_state_atoms->push_back(pack);
			}
		}
	}
}

void compute_bpros_features(const std::vector<int>& basic_features, std::vector<int>& screen_state_atoms) {
	std::pair<std::pair<size_t, size_t>, int> f1, f2;
	for (size_t j = 0; j < basic_features.size(); ++j) {
		unpack_basic_feature(basic_features[j], f1);
		for (size_t k = j; k < basic_features.size(); ++k) {
			unpack_basic_feature(basic_features[k], f2);
			int pack = pack_bpros_feature(f1, f2);
			if (!bpros_features_bitmap_[pack - num_basic_features_]) {
				bpros_features_bitmap_[pack - num_basic_features_] = true;
				screen_state_atoms.push_back(pack);
			}
		}
	}
}

void compute_bprot_features(const std::vector<int>& basic_features,
	std::vector<int>& screen_state_atoms,
	const std::vector<int>& prev_screen_state_atoms) {
	std::pair<std::pair<size_t, size_t>, int> f1, f2;
	for (size_t j = 0; j < basic_features.size(); ++j) {
		unpack_basic_feature(basic_features[j], f1);
		for (size_t k = 0; k < prev_screen_state_atoms.size(); ++k) {
			if (!is_basic_feature(prev_screen_state_atoms[k])) break; // no more basic features in vector
			unpack_basic_feature(prev_screen_state_atoms[k], f2);
			int pack = pack_bprot_feature(f1, f2);
			if (!bprot_features_bitmap_[pack - num_basic_features_ - num_bpros_features_]) {
				bprot_features_bitmap_[pack - num_basic_features_ - num_bpros_features_] = true;
				screen_state_atoms.push_back(pack);
			}
		}
	}
}


std::vector<int>* check_vector(std::vector<int> screen, std::vector<int> prev_screen, bool bprost) {
	basic_features_bitmap_ = std::vector<bool>(num_basic_features_, false);
	std::vector<int>* screen_state_atoms = new std::vector<int>();

	for (size_t c = 0; c <= width - 10; c += 10) { // 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
		for (size_t r = 0; r <= height - 15; r += 15) { // 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195
			compute_basic_features(c / 10, r / 15, screen, screen_state_atoms);
		}
	}
	int num_basic_features = screen_state_atoms->size();
	std::vector<int> basic_features(*screen_state_atoms);
	bpros_features_bitmap_ = std::vector<bool>(num_bpros_features_, false);
	compute_bpros_features(basic_features, *screen_state_atoms);
	if (bprost) {
		return screen_state_atoms;
	}
	int num_bpros_features = screen_state_atoms->size();
	bprot_features_bitmap_ = std::vector<bool>(num_bprot_features_, false);
	compute_bprot_features(basic_features, *screen_state_atoms, prev_screen);
	int num_bprot_features = screen_state_atoms->size() - num_basic_features - num_bpros_features;
	return screen_state_atoms;
}