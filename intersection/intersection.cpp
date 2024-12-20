#include "intersection.h"
#include "Tools.h"
#include <cmath>
#include <tuple>

// ============ Producto Punto =============
const float dotProduct(const float a[], float *&b) {
	float c = 0;

	#pragma omp simd reduction(+:c)
	for(uint32_t i=0; i<3; i++)
		c += a[i]*b[i];

	return c;
}

// ============ Producto Cruz ==============
float* crossProduct(const float a[], const float b[]) {
	float *c = new float[3];
	uint32_t i=1, j=2;

	#pragma omp simd
	for(uint32_t k=0; k<3; k++) {
		c[k] = a[i]*b[j] - a[j]*b[i];

		i = (i + 1) % 3;
		j = (j + 1) % 3;
	}

	return c;
}

// =========== Función que calcula la intersección entre un rayo con un triángulo ====================
bool ray_triangle_intersection(const float ray_near[], const float ray_dir[], const float Points[][3], float &t) {


	const float eps = 0.000001;
	float edges[2][3];

	#pragma omp simd collapse(2)
	for(uint32_t i=0; i<2; i++) {
		for(uint32_t j=0; j<3; j++) {
			edges[i][j] = Points[i+1][j] - Points[0][j];
		}
	}

	float *pvec = crossProduct(ray_dir, edges[1]);
	const float det = dotProduct(edges[0], pvec);

	
	if(fabs(det) < eps) {
		delete[] pvec;
		return false;
	}

  	const float inv_det = 1. / det;

  	float tvec[3];
  	#pragma omp simd
  	for(uint32_t i=0; i<3; i++)
  		tvec[i] = ray_near[i] - Points[0][i];

  	
  	const float u = dotProduct(tvec, pvec) * inv_det;
  	delete[] pvec;
  	
  	if((u < 0.) || (u > 1.))
  		return false;
  	

  	float *qvec = crossProduct(tvec, edges[0]);
  	const float v = dotProduct(ray_dir, qvec) * inv_det;

  	if((v < 0.) || (u + v > 1.)) {
  		delete[] qvec;
  		return false;
  	}

  	t = dotProduct(edges[1], qvec) * inv_det;
  	delete[] qvec;

  	if(t < eps)
  		return false;

	return true;
}

float** multiple_vertices(float **&triangle) {

	float **pt = new float*[6];
	for(uint32_t i=0; i<6; i++)
		pt[i] = new float[3];

	for(uint32_t i=0; i<3; i++) {
		pt[0][i] = triangle[0][i];
		pt[2][i] = triangle[1][i];
		pt[4][i] = triangle[2][i];

		pt[1][i] = (triangle[0][i] + triangle[1][i]) / 2.0;
		pt[3][i] = (triangle[1][i] + triangle[2][i]) / 2.0;
		pt[5][i] = (triangle[0][i] + triangle[2][i]) / 2.0;
	}

    delete[] triangle;
    return pt;
}

float*** multiple_triangles(float ***&triangles, uint32_t &len, const uint32_t polys[][3]) {
	float ***new_triangles = new float**[len*4];

	for(uint32_t i=0; i<len; i++) {
		float **tri = multiple_vertices(triangles[i]);

		for(uint32_t j=0; j<4; j++) {
			new_triangles[i*4 + j] = new float*[3];

			for(uint32_t k=0; k<3; k++) {
				new_triangles[i*4 + j][k] = new float[3];

				for(uint32_t l=0; l<3; l++)
					new_triangles[i*4 + j][k][l] = tri[polys[j][k]][l];
			}
		}

		for(uint32_t j=0; j<6; j++)
			delete[] tri[j];
		delete[] tri;
	}
	len = len * 4;
	delete[] triangles;
	return new_triangles;
}

float** triangle_interpolation(float ***&triangles, const uint32_t &N) {
	float **centroid = new float*[int(pow(4,N))];
	uint32_t len = 1;
	uint32_t polys[4][3] = {{0,1,5},{1,2,3},{5,3,4},{1,3,5}};

	for(uint32_t i=0; i<N; i++)
		triangles = multiple_triangles(triangles, len, polys);

	for(uint32_t i=0; i<len; i++) {
		centroid[i] = new float[3];

		for(uint32_t j=0; j<3; j++) {
			float sum = 0;

			#pragma omp simd reduction(+:sum)
			for(uint32_t k=0; k<3; k++)
				sum += triangles[i][k][j];

			centroid[i][j] = sum / 3.0;
		}
		for(uint32_t j=0; j<3; j++)
			delete[] triangles[i][j];
		delete[] triangles[i];
	}

	delete[] triangles;
	return centroid;
}

const bool getMeshAndFiberEndIntersection(float *&fiberP0, float *&fiberP1, const uint32_t &nPoints, const uint32_t &nPtsLine, const uint32_t &N, const uint32_t &npbp,
	float **&index, const float &step, bool ***&cubeNotEmpty, const std::vector<std::vector<std::vector<std::vector<uint32_t>>>> &centroidIndex,
	const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> &almacen, float **&vertex, uint32_t **&polygons, uint32_t &Ind, float *&ptInt, uint32_t indice_fibra, char hemi, uint32_t *counts) {
	//std::cout << "Indice fibra: " << indice_fibra << std::endl;
    float dd[3];
	int flag_negativo = 0;
	

    #pragma omp simd
    for(uint32_t i=0; i<3; i++){
    	dd[i] = (fiberP0[i] - fiberP1[i]) / float(npbp);
	}

    std::vector<std::vector<uint32_t>> indexes;

    for(uint32_t i=0; i <= nPtsLine*npbp + npbp; i++) {

    	uint32_t I[3];

		//std::cout << "I[] antes" << std::endl;
    	#pragma omp simd
    	for(uint32_t j=0; j<3; j++){
    		I[j] = ( (fiberP1[j] + i * dd[j]) - index[j][0] ) / step;
			//std::cout << I[j] << " ";
			if(I[j] > 1000000 || I[j] < 0){
				//std::cout << "Hemisferio: " << hemi << ", Primer término: " << (fiberP1[j] + i * dd[j]) << std::endl;

				if (hemi == 'R'){
					//std::cout << "Fibra RH" << std::endl;
					flag_negativo++;
				}
				
				else if(hemi == 'L'){
					//std::cout << "Fibra LH" << std::endl;
					flag_negativo++;
				}
			}
		}
		//std::cout << "I[] despues" << std::endl;

		//Si I[j] da un valor negativo, significa que ese extremo no pertenece al hemisferio actual.
		if(flag_negativo !=0){
			//std::cout << "FALSE" << std::endl;
			return false;
		}


		//std::cout << "CUBENOTEMPTY" <<std::endl;
        #pragma omp simd collapse(3)
        for(int32_t a=-1; a<2; a++) {
        	for(int32_t b=-1; b<2; b++) {
        		for(int32_t c=-1; c<2; c++) {
					if((I[0]+a) > 2000000000 || (I[1]+b) > 2000000000 || (I[2]+c) > 2000000000 || (I[0]+a) < 0 || (I[1]+b) < 0 || (I[2]+c) < 0){
						flag_negativo++;
						continue;
					}
					else if( (I[0]+a) >= counts[0] || (I[1]+b) >= counts[1] ||  (I[2]+c) >= counts[2]){
						continue;
					}

        			else if(cubeNotEmpty[ I[0]+a ][ I[1]+b ][ I[2]+c ]) {
        				std::vector<uint32_t> INDEX(3);

        				int32_t abc[3] = {a, b, c};

        				for(uint32_t k=0; k<3; k++)
        					INDEX[k] = I[k] + abc[k];

        				indexes.emplace_back(INDEX);
        			}
        		}
        	}
        }
    }
    
    if(indexes.empty())
    	return false;

	else if(flag_negativo !=0){
		return false;
	}

    std::sort(indexes.begin(), indexes.end());
    indexes.erase( std::unique( indexes.begin(), indexes.end() ), indexes.end() );
    std::vector<std::vector<double>> listDist;

    for(const std::vector<uint32_t>& I : indexes) {
    	for(uint32_t u=0; u<centroidIndex[I[0]][I[1]][I[2]].size(); u++) {
    		double cen[3], dist = 0;

    		#pragma omp simd reduction(+:dist)
    		for(uint32_t i=0; i<3; i++) {
    			cen[i] = almacen[I[0]][I[1]][I[2]][u][i];
    			dist += fabs(cen[i] - fiberP0[i]);
    		}

    		const uint32_t& c_index = centroidIndex[I[0]][I[1]][I[2]][u];
    		listDist.emplace_back((std::vector<double>){dist, (double)c_index});
    	}
    }
    std::sort(listDist.begin(), listDist.end());
    std::vector<uint32_t> listIndex;
    ptInt = new float[3];
    
	for(const std::vector<double>& ind : listDist) {
		if(std::find(listIndex.begin(), listIndex.end(), (uint32_t)ind[1]) == listIndex.end())
			listIndex.emplace_back((uint32_t)ind[1]);

		else
			continue;

		float ray_dir[3], ray_near[3];

		#pragma omp simd
		for(uint32_t i=0; i<3; i++) {
			ray_dir[i] = fiberP1[i] - fiberP0[i];
			ray_near[i] = fiberP1[i];
		}
        
        uint32_t *Triangle = polygons[(uint32_t)ind[1]];
        float Pts[3][3];

        #pragma omp simd collapse(2)
        for(uint32_t i=0; i<3; i++) {
        	for(uint32_t j=0; j<3; j++) {
        		Pts[i][j] = vertex[Triangle[i]][j];
        	}
        }

		float t;

		// ========== Verifica la intersección entre el rayo y el triángulo =============
        if(ray_triangle_intersection(ray_near, ray_dir, Pts, t)) {
	       
        	#pragma omp simd
        	for(uint32_t i=0; i<3; i++)
        		ptInt[i] = fiberP1[i] + (fiberP1[i] - fiberP0[i])*t;	//Punto de intersección creo.

        	Ind = (uint32_t)ind[1];		//Indice del triángulo creo.

        	return true;
        }
       
        
        else {
        	// ============= Ídem, pero con el rayo apuntando hacia el sentido contrario =============
	        

        	float ray_invert[3];
        	#pragma omp simd
        	for(uint32_t i=0; i<3; i++)
        		ray_invert[i] = -ray_dir[i];

            if(ray_triangle_intersection(ray_near, ray_invert, Pts, t)) {
            
            	#pragma omp simd
	        	for(uint32_t i=0; i<3; i++)
	        		ptInt[i] = fiberP1[i] - (fiberP1[i] - fiberP0[i])*t;	//Punto de intersección creo.

	        	Ind = (uint32_t)ind[1];		//Indice del triángulo creo.

	        	return true;
            }
        }
	}

	delete[] ptInt;
	return false;
}
//La función anterior es para aplicar la intersección en sentido directo e inverso de un punto extremo

const std::tuple<bool, bool> getMeshAndFiberIntersection(float **&fiber, const uint32_t &nPoints, const uint32_t &nPtsLine, const uint32_t &N, const uint32_t &npbp, float **&index,
	const float &step, bool ***&cubeNotEmpty, const std::vector<std::vector<std::vector<std::vector<uint32_t>>>> &centroidIndex,
	const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> &almacen, float **&vertex, uint32_t **&polygons,
	uint32_t &InInd, uint32_t &FnInd, float *&InPtInt, float *&FnPtInt, uint32_t indice_fibra, char hemi, uint32_t *counts) {

	bool findInt_In;
	bool findInt_Fn;

	findInt_In = getMeshAndFiberEndIntersection(fiber[0], fiber[1], nPoints, nPtsLine, N, npbp, index, step, cubeNotEmpty,
								   			 centroidIndex, almacen, vertex, polygons, InInd, InPtInt, indice_fibra, hemi, counts);



	findInt_Fn = getMeshAndFiberEndIntersection(fiber[nPoints-1], fiber[nPoints-2], nPoints, nPtsLine, N, npbp, index, step, cubeNotEmpty,
								   			 centroidIndex, almacen, vertex, polygons, FnInd, FnPtInt, indice_fibra, hemi, counts);

	
	return std::tuple<bool, bool>{findInt_In, findInt_Fn};
}


//Joaquín
//--------
//se agrega el segundo mallado.
void meshAndBundlesIntersection(float **&vertex, const uint32_t &n_vertex, uint32_t **&polygons, const uint32_t &n_polygons,
	const uint32_t &nBundles, uint32_t *&nFibers, uint32_t **&nPoints, float ****&Points, const uint32_t& nPtsLine,
	std::vector<std::vector<uint32_t>> &InTri, std::vector<std::vector<uint32_t>> &FnTri, std::vector<std::vector<std::vector<float>>> &InPoints,
	std::vector<std::vector<std::vector<float>>> &FnPoints, std::vector<std::vector<uint32_t>> &fib_index, std::vector<std::vector<uint32_t>> &hemi_membership, 
	float **&Rvertex, const uint32_t &n_Rvertex, uint32_t **&Rpolygons, const uint32_t &n_Rpolygons, std::vector<std::vector<uint32_t>> &fiber_class) {

	uint32_t N = 1;
	uint32_t RN = 1;	//Joaquín

	//std::cout << "Check 1" << std::endl;
	float mdbv = 0; // maximum distance between vertices
	float Rmdbv = 0; //Joaquín: maximum distance between vertices del mallado rh.

	#pragma omp parallel for schedule(dynamic)
	for(uint32_t i=0; i<n_polygons; i++) {
		//std::cout << "i: " << i << std::endl;
		float **pts = new float*[3];
		pts[0] = vertex[polygons[i][0]];	//obtiene las coordenadas de cada vertice. (en 3D) pt[0] = (x0,y0,z0)
		pts[1] = vertex[polygons[i][1]];
		pts[2] = vertex[polygons[i][2]];
		
		//std::cout << n_polygons << std::endl;
		//std::cout << "n_vertex" << n_vertex << std::endl;
		//std::cout << "Vertice 3D" << i << ": " << vertex[polygons[i][0]] << " ; " << vertex[polygons[i][1]] << " ; " << vertex[polygons[i][0]] << std::endl;

		float dists[3] = {0,0,0};
		for(uint32_t k=0; k<3; k++){
			//std::cout << "Check antes dist" << std::endl;
			//std::cout << "pts[0][k]: " << pts[0][k] << std::endl;
			//std::cout << "pts[1][k]: " << pts[1][k] << std::endl;
			//std::cout << "agregar pow2: " << pow(pts[0][k] - pts[1][k], 2) << std::endl;
			dists[0] += pow(pts[0][k] - pts[1][k], 2);
			//std::cout << "dists[0]: " << dists[0] << std::endl;
		} 
		for(uint32_t k=0; k<3; k++){
			dists[1] += pow(pts[1][k] - pts[2][k], 2);
			//std::cout << "dists[1]: " << dists[1] << std::endl;
		} 
		for(uint32_t k=0; k<3; k++){
			dists[2] += pow(pts[2][k] - pts[0][k], 2);
			//std::cout << "dists[2]: " << dists[2] << std::endl;
		} 
		for(uint32_t k=0; k<3; k++){
			dists[k] = sqrt(dists[k]);	//Distancia entre vertices para mdbv
			//std::cout << "dists[k]: " << dists[k] << std::endl;
		} 

		delete[] pts;

		const float newMax = *std::max_element(std::begin(dists), std::end(dists));
		//std::cout << "New MAX: " << newMax << std::endl;
		#pragma omp critical
		if(newMax > mdbv)
			mdbv = newMax;	//se encuentra el máximo mdbv entre todos los triángulos.
	}

	//std::cout << "Check LH" << std::endl;

	//Joaquín: Lo mismo, pero en rh
	#pragma omp parallel for schedule(dynamic)
	for(uint32_t i=0; i<n_Rpolygons; i++) {
		float **Rpts = new float*[3];
		Rpts[0] = Rvertex[Rpolygons[i][0]];	//obtiene las coordenadas de cada vertice.
		Rpts[1] = Rvertex[Rpolygons[i][1]];
		Rpts[2] = Rvertex[Rpolygons[i][2]];

		float Rdists[3] = {0,0,0};
		for(uint32_t k=0; k<3; k++) Rdists[0] += pow(Rpts[0][k] - Rpts[1][k], 2);
		for(uint32_t k=0; k<3; k++) Rdists[1] += pow(Rpts[1][k] - Rpts[2][k], 2);
		for(uint32_t k=0; k<3; k++) Rdists[2] += pow(Rpts[2][k] - Rpts[0][k], 2);
		for(uint32_t k=0; k<3; k++) Rdists[k] = sqrt(Rdists[k]);	//Distancia entre vertices para mdbv

		delete[] Rpts;

		const float RnewMax = *std::max_element(std::begin(Rdists), std::end(Rdists));
		#pragma omp critical
		if(RnewMax > Rmdbv)
			Rmdbv = RnewMax;	//se encuentra el máximo mdbv entre todos los triángulos.
	}

	//std::cout << "Check 2" << std::endl;
	const float step = mdbv / pow(2, N + 1);	//Cálculo de step
	const float Rstep = Rmdbv / pow(2, RN + 1);	//Joaquín: Cálculo de step rh

	//Debe ser el mismo para ambos hemisferios.
	float mdbp = 0; // maximum distance between points
	for(uint32_t i=0; i<nBundles; i++) {
		for(uint32_t j=0; j<nFibers[i]; j++) {

			float dist = 0;

			for(uint32_t k=0; k<3; k++)
				dist += pow(Points[i][j][0][k] - Points[i][j][1][k], 2);

			dist = sqrt(dist);
			if(dist > mdbp)
				mdbp = dist;

			dist = 0;
			for(uint32_t k=0; k<3; k++)
				dist += pow(Points[i][j][0][k] - Points[i][j][1][k], 2);

			dist = sqrt(dist);
			if(dist > mdbp)
				mdbp = dist;
		}
	}

	uint32_t npbp = mdbp / step; // number of points between points
	uint32_t Rnpbp = mdbp / Rstep; //Joaquín: number of points between points rh... mdbp el mismo para ambos hemisferios, pero los steps cambian


	std::vector<float> vx(n_vertex);
	std::vector<float> vy(n_vertex);
	std::vector<float> vz(n_vertex);
	// ===== Find outermost vertex ===========
	#pragma omp parallel for schedule(dynamic)
	for(uint32_t i=0; i<n_vertex; i++) {
		vx[i] = vertex[i][0];	//Coordenada x del i-ésimo triángulo.
		vy[i] = vertex[i][1];	//Coordenada y del i-ésimo triángulo.
		vz[i] = vertex[i][2];	//Coordenada z del i-ésimo triángulo.
	}
	
	//Ordena las coordenadas de menor a mayor.
	std::sort(vx.begin(), vx.end()); 
	std::sort(vy.begin(), vy.end());
	std::sort(vz.begin(), vz.end());

	//Límites espaciales (eq. 3.6 y 3.7 MT Felipe)
	const float minx = *vx.begin()  - (nPtsLine + 1) * mdbp - 4 * step;		   // coordenada x minima
	const float maxx = *vx.rbegin() + (nPtsLine + 1) * mdbp + 4 * step;		   // coordenada x maxima
	const float miny = *vy.begin()  - (nPtsLine + 1) * mdbp - 4 * step;		   // coordenada y minima
	const float maxy = *vy.rbegin() + (nPtsLine + 1) * mdbp + 4 * step;		   // coordenada y maxima
	const float minz = *vz.begin()  - (nPtsLine + 1) * mdbp - 4 * step;		   // coordenada z minima
	const float maxz = *vz.rbegin() + (nPtsLine + 1) * mdbp + 4 * step;		   // coordenada z maxima

	
	//Joaquín: Lo mismo para rh
	//-------
	std::vector<float> Rvx(n_Rvertex);
	std::vector<float> Rvy(n_Rvertex);
	std::vector<float> Rvz(n_Rvertex);
	// ===== Find outermost vertex ===========
	#pragma omp parallel for schedule(dynamic)
	for(uint32_t i=0; i<n_Rvertex; i++) {
		Rvx[i] = Rvertex[i][0];	//Coordenada x del i-ésimo triángulo.
		Rvy[i] = Rvertex[i][1];	//Coordenada y del i-ésimo triángulo.
		Rvz[i] = Rvertex[i][2];	//Coordenada z del i-ésimo triángulo.
	}
	
	//Ordena las coordenadas de menor a mayor.
	std::sort(Rvx.begin(), Rvx.end()); 
	std::sort(Rvy.begin(), Rvy.end());
	std::sort(Rvz.begin(), Rvz.end());

	//Límites espaciales (eq. 3.6 y 3.7 MT Felipe)
	const float Rminx = *Rvx.begin()  - (nPtsLine + 1) * mdbp - 4 * Rstep;		   // coordenada x minima
	const float Rmaxx = *Rvx.rbegin() + (nPtsLine + 1) * mdbp + 4 * Rstep;		   // coordenada x maxima
	const float Rminy = *Rvy.begin()  - (nPtsLine + 1) * mdbp - 4 * Rstep;		   // coordenada y minima
	const float Rmaxy = *Rvy.rbegin() + (nPtsLine + 1) * mdbp + 4 * Rstep;		   // coordenada y maxima
	const float Rminz = *Rvz.begin()  - (nPtsLine + 1) * mdbp - 4 * Rstep;		   // coordenada z minima
	const float Rmaxz = *Rvz.rbegin() + (nPtsLine + 1) * mdbp + 4 * Rstep;		   // coordenada z maxima

	//std::cout << "Límites espaciales check" << std::endl;
	std::vector<std::vector<float**>> Bundles;
	std::vector<std::vector<uint32_t>> new_nBundles;
	
	for(uint32_t i=0; i<nBundles; i++) {
		std::vector<float**> new_Points;
		std::vector<uint32_t> new_nPoints;

		//std::cout << nFibers[i] << std::endl;

		int contador_previous_fibers_test = 0;
		int contador_if_test = 0;

		for(uint32_t j=0; j<nFibers[i]; j++) {
			
			contador_previous_fibers_test +=1;
			//Para cada fibra de cada fascículo, se verifica que los puntos extremos (1ro y último) estén dentro de el mallado, más una tolerancia dada por +- mdbp +- 2*step

			//mallado lh
			const bool exi = ((*vx.begin() - mdbp - 2*step) <= Points[i][j][0][0]) && (Points[i][j][0][0] <= (*vx.rbegin() + mdbp + 2*step));
			const bool eyi = ((*vy.begin() - mdbp - 2*step) <= Points[i][j][0][1]) && (Points[i][j][0][1] <= (*vy.rbegin() + mdbp + 2*step));
			const bool ezi = ((*vz.begin() - mdbp - 2*step) <= Points[i][j][0][2]) && (Points[i][j][0][2] <= (*vz.rbegin() + mdbp + 2*step));

			//Joaquin:
			//std::cout << nPoints[i][j]-1 << std::endl;
			//nPoints[i][j]-1 es igual a 20
			//Estos son los puntos finales de la fibra. Los anteriores era para los primeros puntos extremos. 
			//Ver que estén dentro es de la siguiente forma...
							//(limites espaciales - mdbp - 2*step  <=  punto extremo de la fibra )  &&  (punto extremo de la fibra <=  limites espaciales + mdbp + 2*step) 

			const bool exf = ((*vx.begin() - mdbp - 2*step) <= Points[i][j][nPoints[i][j]-1][0]) && (Points[i][j][nPoints[i][j]-1][0] <= (*vx.rbegin() + mdbp + 2*step));
			const bool eyf = ((*vy.begin() - mdbp - 2*step) <= Points[i][j][nPoints[i][j]-1][1]) && (Points[i][j][nPoints[i][j]-1][1] <= (*vy.rbegin() + mdbp + 2*step));
			const bool ezf = ((*vz.begin() - mdbp - 2*step) <= Points[i][j][nPoints[i][j]-1][2]) && (Points[i][j][nPoints[i][j]-1][2] <= (*vz.rbegin() + mdbp + 2*step));

			//Joaquin:
			//La versión anterior requería que AMBOS putnos extremos estén dentro de los limites espaciales.
			//En esta versión necesitamos que estén dentro de los límites espaciales, pero de alguno de los mallados...
			//No es necesario que los puntos extremos estén dentro del mismo mallado, solo que los puntos extremos esté por lo menos en uno de ellos...
			//por lo que la condicion se cambia a: que el punto inicial esté dentro de mallado LH o RH Y que el final esté dentro de LH o RH.
			
			//joaquin:
			//mallado rh
			const bool Rexi = ((*Rvx.begin() - mdbp - 2*Rstep) <= Points[i][j][0][0]) && (Points[i][j][0][0] <= (*Rvx.rbegin() + mdbp + 2*Rstep));
			const bool Reyi = ((*Rvy.begin() - mdbp - 2*Rstep) <= Points[i][j][0][1]) && (Points[i][j][0][1] <= (*Rvy.rbegin() + mdbp + 2*Rstep));
			const bool Rezi = ((*Rvz.begin() - mdbp - 2*Rstep) <= Points[i][j][0][2]) && (Points[i][j][0][2] <= (*Rvz.rbegin() + mdbp + 2*Rstep));

			const bool Rexf = ((*Rvx.begin() - mdbp - 2*Rstep) <= Points[i][j][nPoints[i][j]-1][0]) && (Points[i][j][nPoints[i][j]-1][0] <= (*Rvx.rbegin() + mdbp + 2*Rstep));
			const bool Reyf = ((*Rvy.begin() - mdbp - 2*Rstep) <= Points[i][j][nPoints[i][j]-1][1]) && (Points[i][j][nPoints[i][j]-1][1] <= (*Rvy.rbegin() + mdbp + 2*Rstep));
			const bool Rezf = ((*Rvz.begin() - mdbp - 2*Rstep) <= Points[i][j][nPoints[i][j]-1][2]) && (Points[i][j][nPoints[i][j]-1][2] <= (*Rvz.rbegin() + mdbp + 2*Rstep));

			//nuevo if
			if(1){//((exi && eyi && ezi) || (Rexi && Reyi && Rezi)) && ((exf && eyf && ezf) || (Rexf && Reyf && Rezf))) {
				new_Points.emplace_back(Points[i][j]);
				new_nPoints.emplace_back(nPoints[i][j]);
				contador_if_test +=1;
			}

		}
		bool test = (contador_previous_fibers_test == contador_if_test);
		//std::cout << test << '\t' << contador_previous_fibers_test << '\t' << contador_if_test << std::endl;

		//Fascículos luego de filtrar por aquellas fibras que están dentro de los límites.
		Bundles.emplace_back(new_Points);
		new_nBundles.emplace_back(new_nPoints);
	}
	//std::cout << "Check fibras límites espaciales" << std::endl;

	//Cálculo de nueva cantidad de fibras en cada fascículo.
	for(uint32_t i=0; i<nBundles; i++) {
		nFibers[i] = Bundles[i].size();
		delete[] Points[i];
		delete[] nPoints[i];

		//std::cout << Bundles[i].size() << std::endl;

		//Puntos de fibras nuevas luego del filtrado inicial.
		Points[i] = new float**[nFibers[i]];
		nPoints[i] = new uint32_t[nFibers[i]];

		for(uint32_t j=0; j<nFibers[i]; j++) {
			Points[i][j] = Bundles[i][j];
			nPoints[i][j] = new_nBundles[i][j];
		}
	}//ok

	//		=====================================================
	//					Left hemisphere
	//		=====================================================

	// ================ Obtiene la cantidad de intervalos por eje ==================
	const float mins[3] = {minx, miny, minz};
	const float maxs[3] = {maxx, maxy, maxz};
	uint32_t counts[3] = {0, 0, 0};		//Joaquín: cantidad de cubos para cada eje, según el min-max (x,y,z)

	//Cuenta cuántos cubos (N_cubos) son necesarios para cubrir los límites espaciales, en cada coordenada (desde el límite inferior hasta el límite superior).
	for(uint32_t i=0; i<3; i++) {
		float ini = mins[i];
		while(ini < maxs[i]) {
			counts[i]++;
			ini += step;
		}//Se aumenta 1 el contador hasta que se llegue al máximo.
	}

	// ====== Generacion de intervalos (coordenadas de los vertices de cada cubo) ===================
	float** index = new float*[3];

	for(uint32_t i=0; i<3; i++) {
		index[i] = new float[counts[i] + 1];	//Reserva un arreglo con N_cubos elementos, para cada coordenada (x,y,z).

		#pragma omp simd
		for(uint32_t j=0; j<counts[i] + 1; j++) {
			index[i][j] = mins[i] + j * step;	//Guarda las coordenadas de cada cubo, en cada coordenada (x,y,z).
		}
	}

	std::vector<std::vector<float>> centroids(n_polygons*pow(4,N), std::vector<float>(3));	//Vector para almacenar los centroides de cada triángulo.

	#pragma omp parallel for schedule(dynamic)
	for(uint32_t i=0; i<n_polygons; i++) {
		float ***triangles = new float**[1];
		triangles[0] = new float*[3];

		for(uint32_t j=0; j<3; j++)
			triangles[0][j] = vertex[polygons[i][j]];

		//Obtiene los centroides de cada triángulo.
		float **centroid = triangle_interpolation(triangles, N);

		for(uint32_t j=0; j<pow(4,N); j++) {
			for(uint32_t k=0; k<3; k++)
				centroids[i*pow(4,N) + j][k] = centroid[j][k];
			delete[] centroid[j];
		}
		delete[] centroid;
	}//ok, se almacenan los centroides en 'centroids'.

	//Vectores que almacenan cada cubo y cada centroide.
	std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> almacen;
	std::vector<std::vector<std::vector<std::vector<uint32_t>>>> centroidIndex;
	bool ***cubeNotEmpty = new bool**[counts[0]];

	//Se cambian al tamaño de N_cubos en cada coordenada (x,y,z).
	almacen.resize(counts[0]);
	centroidIndex.resize(counts[0]);

	//Cada cubo se rellena con "False", es decir, está vacío.
	for(uint32_t ix = 0; ix < counts[0]; ix++) {

		almacen[ix].resize(counts[1]);
		centroidIndex[ix].resize(counts[1]);
		cubeNotEmpty[ix] = new bool*[counts[1]];

		for(uint32_t iy = 0; iy < counts[1]; iy++) {

			almacen[ix][iy].resize(counts[2]);
			centroidIndex[ix][iy].resize(counts[2]);
			cubeNotEmpty[ix][iy] = new bool[counts[2]];

			for(uint32_t iz = 0; iz < counts[2]; iz++) {
				cubeNotEmpty[ix][iy][iz] = false;
			}
		}
	}


	//		=====================================================
	//					Right hemisphere
	//		=====================================================

	// ================ Obtiene la cantidad de intervalos por eje ==================
	const float Rmins[3] = {Rminx, Rminy, Rminz};
	const float Rmaxs[3] = {Rmaxx, Rmaxy, Rmaxz};
	uint32_t Rcounts[3] = {0, 0, 0};		//Joaquín: cantidad de cubos para cada eje, según el min-max (x,y,z)

	//Cuenta cuántos cubos (N_cubos) son necesarios para cubrir los límites espaciales, en cada coordenada (desde el límite inferior hasta el límite superior).
	for(uint32_t i=0; i<3; i++) {
		float Rini = Rmins[i];
		while(Rini < Rmaxs[i]) {
			Rcounts[i]++;
			Rini += Rstep;
		}//Se aumenta 1 el contador hasta que se llegue al máximo.
	}

	// ====== Generacion de intervalos (coordenadas de los vertices de cada cubo) ===================
	float** Rindex = new float*[3];

	for(uint32_t i=0; i<3; i++) {
		Rindex[i] = new float[Rcounts[i] + 1];	//Reserva un arreglo con N_cubos elementos, para cada coordenada (x,y,z).

		#pragma omp simd
		for(uint32_t j=0; j<Rcounts[i] + 1; j++) {
			Rindex[i][j] = Rmins[i] + j * Rstep;	//Guarda las coordenadas de cada cubo, en cada coordenada (x,y,z).
		}
	}

	std::vector<std::vector<float>> Rcentroids(n_Rpolygons*pow(4,RN), std::vector<float>(3));	//Vector para almacenar los centroides de cada triángulo.

	#pragma omp parallel for schedule(dynamic)
	for(uint32_t i=0; i<n_Rpolygons; i++) {
		float ***Rtriangles = new float**[1];
		Rtriangles[0] = new float*[3];

		for(uint32_t j=0; j<3; j++)
			Rtriangles[0][j] = Rvertex[Rpolygons[i][j]];

		//Obtiene los centroides de cada triángulo.
		float **Rcentroid = triangle_interpolation(Rtriangles, RN);

		for(uint32_t j=0; j<pow(4,RN); j++) {
			for(uint32_t k=0; k<3; k++)
				Rcentroids[i*pow(4,RN) + j][k] = Rcentroid[j][k];
			delete[] Rcentroid[j];
		}
		delete[] Rcentroid;
	}//ok, se almacenan los centroides en 'centroids'.

	//Vectores que almacenan cada cubo y cada centroide.
	std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> Ralmacen;
	std::vector<std::vector<std::vector<std::vector<uint32_t>>>> RcentroidIndex;
	bool ***RcubeNotEmpty = new bool**[Rcounts[0]];

	//Se cambian al tamaño de N_cubos en cada coordenada (x,y,z).
	Ralmacen.resize(Rcounts[0]);
	RcentroidIndex.resize(Rcounts[0]);

	//Cada cubo se rellena con "False", es decir, está vacío.
	for(uint32_t ix = 0; ix < Rcounts[0]; ix++) {

		Ralmacen[ix].resize(Rcounts[1]);
		RcentroidIndex[ix].resize(Rcounts[1]);
		RcubeNotEmpty[ix] = new bool*[Rcounts[1]];

		for(uint32_t iy = 0; iy < Rcounts[1]; iy++) {

			Ralmacen[ix][iy].resize(Rcounts[2]);
			RcentroidIndex[ix][iy].resize(Rcounts[2]);
			RcubeNotEmpty[ix][iy] = new bool[Rcounts[2]];

			for(uint32_t iz = 0; iz < Rcounts[2]; iz++) {
				RcubeNotEmpty[ix][iy][iz] = false;
				//std::cout << ix << " " << iy << " " << iz << ", Valor: " << RcubeNotEmpty[ix][iy][iz] << std::endl;
			}
		}
	}
	//std::cout << "Check cubos" << std::endl;

	//***LH****
	// =========================== En cada cubo almacena una lista de centroides y su indice ======================

	for(uint32_t i=0; i<n_polygons*pow(4,N); i++) {

		uint32_t I[3];

		#pragma omp simd
		for(uint32_t j=0; j<3; j++)
			I[j] = (centroids[i][j] - index[j][0]) / step;

		almacen[I[0]][I[1]][I[2]].emplace_back(centroids[i]);
		centroidIndex[I[0]][I[1]][I[2]].emplace_back(uint32_t(i / pow(4,N)));
		cubeNotEmpty[I[0]][I[1]][I[2]] = true;
	}

	//***RH****
	// =========================== En cada cubo almacena una lista de centroides y su indice ======================

	for(uint32_t i=0; i<n_Rpolygons*pow(4,RN); i++) {

		uint32_t RI[3];

		#pragma omp simd
		for(uint32_t j=0; j<3; j++)
			RI[j] = (Rcentroids[i][j] - Rindex[j][0]) / Rstep;

		Ralmacen[RI[0]][RI[1]][RI[2]].emplace_back(Rcentroids[i]);
		RcentroidIndex[RI[0]][RI[1]][RI[2]].emplace_back(uint32_t(i / pow(4,RN)));
		RcubeNotEmpty[RI[0]][RI[1]][RI[2]] = true;
	}

	// ===============================================================================================================

	//Creo que esto debe quedar asi, no para RH y LH... aunque hay que crear algunas variables, como para identificar con que hemisferio intersectan.
	InTri.resize(nBundles);
	FnTri.resize(nBundles);
	InPoints.resize(nBundles);
	FnPoints.resize(nBundles);
	fib_index.resize(nBundles);
	hemi_membership.resize(nBundles);
	fiber_class.resize(nBundles);

	//std::cout << "Check cubos y centroides, listo para intersección" << std::endl;
	
	//std::cout << "RCounts: " << Rcounts[0] << " " << Rcounts[1] << " " << Rcounts[2] << std::endl;
	//std::cout << "LCounts: " << counts[0]  << " " << counts[1]  << " " << counts[2]  << std::endl;
	for(uint32_t i=0; i<nBundles; i++) {

		std::cout << "Bundle: " << (i+1) << "/" << nBundles;
		std::cout << ", Number of fibers: " << nFibers[i] << std::endl;

		//std::cout << "\nConstantes:\n" << "step_lh: " << step << "\nstep_rh: " << Rstep <<  "\nN_lh: " << N << "\nN_rh: " << RN << "\nnpbp_lh: " << npbp << "\nnpbp_rh: " << Rnpbp << std::endl;
		//std::cout << "n_polygons_rh: " << n_Rpolygons << "\nn_polygons_lh: " << n_polygons << "\nmdbv_lh: " << mdbv << "\nmdbv_rh: " << Rmdbv << "\nmdbp: " << mdbp << "\n"<< std::endl;

		std::vector<uint32_t> membership_flags;
		std::vector<uint32_t> listFibInd;
		std::vector<uint32_t> listFibClass;
		std::vector<std::vector<uint32_t>> listTri;
		std::vector<std::vector<std::vector<float>>> listPtInt;

		uint32_t intra_sum = 0;
		uint32_t inter_sum = 0;

		#pragma omp parallel for schedule(dynamic)
		for(uint32_t j=0; j<nFibers[i]; j++) {
			//std::cout << j << std::endl;			

			bool findInt_In, findInt_Fn, findInt_In_lh, findInt_Fn_lh, findInt_In_rh, findInt_Fn_rh; // find intersection, modificado joaquin...
			uint32_t InT_lh, FnT_lh, InT_rh, FnT_rh; // initial and final triangle
			float *InPtInt_lh, *FnPtInt_lh, *InPtInt_rh, *FnPtInt_rh; // Initial and Final point intersection

			//Joaquín:
			char mallado_In, mallado_Fn;

			std::tie(findInt_In_lh, findInt_Fn_lh) = getMeshAndFiberIntersection(Points[i][j], nPoints[i][j], nPtsLine, N, npbp, index, step, cubeNotEmpty,
				 						    centroidIndex, almacen, vertex, polygons, InT_lh, FnT_lh, InPtInt_lh, FnPtInt_lh, j , 'L', counts);


			std::tie(findInt_In_rh, findInt_Fn_rh) = getMeshAndFiberIntersection(Points[i][j], nPoints[i][j], nPtsLine, RN, Rnpbp, Rindex, Rstep, RcubeNotEmpty,
				 						    RcentroidIndex, Ralmacen, Rvertex, Rpolygons, InT_rh, FnT_rh, InPtInt_rh, FnPtInt_rh, j, 'R', Rcounts);


			//Hay intersección en los dos extremos si y solo si hay intersección en alguno de los mallados.
			findInt_In = findInt_In_lh || findInt_In_rh;
			findInt_Fn = findInt_Fn_lh || findInt_Fn_rh;
			
			std::vector<uint32_t> error_fibers;	//fibras q no intersectan

			//Guardar la intersección dependiendo del punto que intersecta y del mallado.
			if(findInt_In && findInt_Fn){
				//std::cout << "Hay intersección" << std::endl;

				//lh-rh
				if(findInt_In_lh && findInt_Fn_rh){

					#pragma omp critical
					inter_sum++;
					#pragma omp critical
					listFibInd.emplace_back(j);
					#pragma omp critical
					listFibClass.emplace_back(3);
					#pragma omp critical
					listTri.emplace_back((std::vector<uint32_t>){InT_lh, FnT_rh});

					#pragma omp critical
					listPtInt.emplace_back((std::vector<std::vector<float>>){{InPtInt_lh[0], InPtInt_lh[1], InPtInt_lh[2]}, {FnPtInt_rh[0], FnPtInt_rh[1], FnPtInt_rh[2]}});

					delete[] InPtInt_lh;
					delete[] FnPtInt_rh;
				}

				//rh-lh
				else if(findInt_In_rh && findInt_Fn_lh){

					#pragma omp critical
					inter_sum++;
					#pragma omp critical
					listFibInd.emplace_back(j);
					#pragma omp critical
					listFibClass.emplace_back(4);
					#pragma omp critical
					listTri.emplace_back((std::vector<uint32_t>){InT_rh, FnT_lh});

					#pragma omp critical
					listPtInt.emplace_back((std::vector<std::vector<float>>){{InPtInt_rh[0], InPtInt_rh[1], InPtInt_rh[2]}, {FnPtInt_lh[0], FnPtInt_lh[1], FnPtInt_lh[2]}});

					delete[] InPtInt_rh;
					delete[] FnPtInt_lh;
				}
				
				//lh-lh
				else if(findInt_In_lh && findInt_Fn_lh){
	
					#pragma omp critical
					intra_sum++;
					#pragma omp critical
					listFibInd.emplace_back(j);
					#pragma omp critical
					listFibClass.emplace_back(1);
					#pragma omp critical
					listTri.emplace_back((std::vector<uint32_t>){InT_lh, FnT_lh});

					#pragma omp critical
					listPtInt.emplace_back((std::vector<std::vector<float>>){{InPtInt_lh[0], InPtInt_lh[1], InPtInt_lh[2]}, {FnPtInt_lh[0], FnPtInt_lh[1], FnPtInt_lh[2]}});

					delete[] InPtInt_lh;
					delete[] FnPtInt_lh;
				}

				//rh-rh
				else if(findInt_In_rh && findInt_Fn_rh){

					#pragma omp critical
					intra_sum++;
					#pragma omp critical
					listFibInd.emplace_back(j);
					#pragma omp critical
					listFibClass.emplace_back(2);
					#pragma omp critical
					listTri.emplace_back((std::vector<uint32_t>){InT_rh, FnT_rh});

					#pragma omp critical
					listPtInt.emplace_back((std::vector<std::vector<float>>){{InPtInt_rh[0], InPtInt_rh[1], InPtInt_rh[2]}, {FnPtInt_rh[0], FnPtInt_rh[1], FnPtInt_rh[2]}});

					delete[] InPtInt_rh;
					delete[] FnPtInt_rh;
				}



			}

			else{
				error_fibers.emplace_back(j);
			}


		}
		
		#pragma omp critical
		if(intra_sum == 0 && inter_sum == 0){
			membership_flags.emplace_back(0);
			membership_flags.emplace_back(0);
		}

		else if(intra_sum >= inter_sum){
			membership_flags.emplace_back(1);
			membership_flags.emplace_back(1);
		}

		else if(intra_sum < inter_sum){
			membership_flags.emplace_back(1);
			membership_flags.emplace_back(0);

		}

		for(uint32_t j=0; j<listTri.size(); j++) {
			//std::cout << listTri[j][1] << std::endl;
			InTri[i].emplace_back(listTri[j][0]);
			FnTri[i].emplace_back(listTri[j][1]);
			InPoints[i].emplace_back(listPtInt[j][0]);
			FnPoints[i].emplace_back(listPtInt[j][1]);
		}

		fib_index[i] = listFibInd;
		fiber_class[i] = listFibClass;
		hemi_membership[i] = membership_flags;
		//std::cout << "\n";
	}

	// =================================================================================================================
	
	// ==========================================================================================================


	for(uint32_t i=0; i<3; i++)
		delete[] index[i];
	delete[] index;

	for(uint32_t i=0; i<counts[0]; i++) {
		for(uint32_t j=0; j<counts[1]; j++) {
			delete[] cubeNotEmpty[i][j];
		}
		delete[] cubeNotEmpty[i];
	}
	delete[] cubeNotEmpty;

	for(uint32_t i=0; i<nBundles; i++) {
		for(uint32_t j=0; j<nFibers[i]; j++) {
			for(uint32_t k=0; k<nPoints[i][j]; k++)
				delete[] Points[i][j][k];
			delete[] Points[i][j];
		}
		delete[] Points[i];
		delete[] nPoints[i];
	}
	
	delete[] Points;
	delete[] nPoints;
	delete[] nFibers;
}
