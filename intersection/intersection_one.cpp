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

	read_mesh(argv[1], Lvertex, Lpolygons, n_Lvertex, n_Lpolygons);
	//read_mesh(argv[2], Rvertex, Rpolygons, n_Rvertex, n_Rpolygons);

	//std::cout << "Poly: " << n_Lpolygons << std::endl;
	//std::cout << "------------------" << std::endl;
	//std::cout << "Vertex: " << n_Lvertex << std::endl;

	// ========================== Se leen las fibras ======================================
	uint32_t nLBundles, nRBundles;
	uint32_t *nLFibers, *nRFibers;
	uint32_t **nLPoints, **nRPoints;
	float ****LPoints, ****RPoints;

	read_bundles(argv[2], nLBundles, nLFibers, nLPoints, LPoints);
	//read_bundles(argv[4], nRBundles, nRFibers, nRPoints, RPoints);
	
	/*
	bool i_flag = (argv[9][0] == '1');
	bool m_flag = (argv[10][0] == '1');
	bool r_flag = (argv[11][0] == '1');	

	if(r_flag){
		for(uint32_t i=0; i<nLBundles; i++){
			for(uint32_t j=0; j<nLFibers[i]; j++){
				std::reverse(LPoints[i][j], LPoints[i][j] + nLPoints[i][j]);
			}
		}
	}
	*/	
	// =========================== Intersección ===========================================
	const uint32_t nPtsLine = 2;

	std::vector<std::vector<uint32_t>> Lfib_index, Rfib_index;
	std::vector<std::vector<uint32_t>> LInTri, LFnTri, RInTri, RFnTri;
	std::vector<std::vector<std::vector<float>>> LInPoints, LFnPoints, RInPoints, RFnPoints;
	std::vector<std::vector<uint32_t>> Lhemi_membership;


	meshAndBundlesIntersection(Lvertex, n_Lvertex, Lpolygons, n_Lpolygons, nLBundles, nLFibers,
							   nLPoints, LPoints, nPtsLine, LInTri, LFnTri, LInPoints, LFnPoints, Lfib_index, Lhemi_membership);

	//meshAndBundlesIntersection(Rvertex, n_Rvertex, Rpolygons, n_Rpolygons, nRBundles, nRFibers,
	//						   nRPoints, RPoints, nPtsLine, RInTri, RFnTri, RInPoints, RFnPoints, Rfib_index, argv[4], argv[8]);

	//std::cout << Lhemi_membership[0][0] << "," << Lhemi_membership[0][1] << std::endl;

	

	//if(i_flag)
	write_intersection(argv[5], argv[3], LInTri, LFnTri, LInPoints, LFnPoints, Lfib_index);
	
	//write_intersection(argv[6], argv[4], RInTri, RFnTri, RInPoints, RFnPoints, Rfib_index);

	//std::cout << Lhemi_membership.size() << std::endl;

	//if(m_flag)
		//write_membership(argv[7], argv[3], Lhemi_membership);	

	//Delete(Lvertex, Lpolygons, n_Lvertex, n_Lpolygons);
	//Delete(Rvertex, Rpolygons, n_Rvertex, n_Rpolygons);

	return 0;
}
