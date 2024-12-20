#pragma once

#include "dirent.h"
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

void read_mesh(const std::string &filename, float **&vertex, uint32_t **&polygons, uint32_t &len_vertex, uint32_t &len_polygons);
void read_bundles(const std::string &path, uint32_t &nBundles, uint32_t *&nFibers, uint32_t **&nPoints, float ****&Points);

void write_intersection(const std::string &path, const std::string &pathBundles,
	const std::vector<std::vector<uint32_t>> &InTri, const std::vector<std::vector<uint32_t>> &FnTri,
	const std::vector<std::vector<std::vector<float>>> &InPoints, const std::vector<std::vector<std::vector<float>>> &FnPoints,
	const std::vector<std::vector<uint32_t>> &fib_index, const std::vector<std::vector<uint32_t>> &fiber_class);

void write_membership(const std::string &path, const std::string &pathBundles, const std::vector<std::vector<uint32_t>> &hemi_membership);

void Delete(float **&vertex, uint32_t **&polygons, const uint32_t &len_vertex, const uint32_t &len_polygons);
