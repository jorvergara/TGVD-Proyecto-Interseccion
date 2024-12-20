#include "intersection.h"
#include "Tools.h"
#include <cmath>
#include "iostream"
#include <algorithm>

int main(int argc, char const *argv[])
{
	float **Lvertex, **Rvertex;
	uint32_t **Lpolygons, **Rpolygons;
	uint32_t n_Lvertex, n_Lpolygons;
	uint32_t n_Rvertex, n_Rpolygons;

	// ========================== Se lee el mallado =======================================
	//std::cout << "leer mallados" << std::endl;
	read_mesh(argv[1], Lvertex, Lpolygons, n_Lvertex, n_Lpolygons);
	read_mesh(argv[2], Rvertex, Rpolygons, n_Rvertex, n_Rpolygons);
	//std::cout << "fin lectura mallados" << std::endl;


	// ========================== Se leen las fibras ======================================
	uint32_t nLBundles, nRBundles;
	uint32_t *nLFibers, *nRFibers;
	uint32_t **nLPoints, **nRPoints;
	float ****LPoints, ****RPoints;
	//std::cout << "leer bundles" << std::endl;
	read_bundles(argv[3], nLBundles, nLFibers, nLPoints, LPoints);

		
	// =========================== Intersección ===========================================
	const uint32_t nPtsLine = 2;

	std::vector<std::vector<uint32_t>> Lfib_index, Rfib_index;
	std::vector<std::vector<uint32_t>> fiber_class;
	std::vector<std::vector<uint32_t>> LInTri, LFnTri, RInTri, RFnTri;
	std::vector<std::vector<std::vector<float>>> LInPoints, LFnPoints, RInPoints, RFnPoints;
	std::vector<std::vector<uint32_t>> Lhemi_membership;
	//std::cout << "calcular interseccion" << std::endl;
	meshAndBundlesIntersection(Lvertex, n_Lvertex, Lpolygons, n_Lpolygons, nLBundles, nLFibers,
							   nLPoints, LPoints, nPtsLine, LInTri, LFnTri, LInPoints, LFnPoints, Lfib_index, Lhemi_membership, 
							   Rvertex, n_Rvertex, Rpolygons, n_Rpolygons, fiber_class);
	

	//std::cout << "write intersection" << std::endl;
	write_intersection(argv[4], argv[3], LInTri, LFnTri, LInPoints, LFnPoints, Lfib_index,fiber_class);
	//std::cout << "end" << std::endl;

	//Delete(Lvertex, Lpolygons, n_Lvertex, n_Lpolygons);
	//Delete(Rvertex, Rpolygons, n_Rvertex, n_Rpolygons);

	return 0;
}