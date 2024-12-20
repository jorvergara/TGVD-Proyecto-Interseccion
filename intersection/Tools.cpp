#include "Tools.h"
#include <iostream>
#include <cmath>
#include <sstream>

// función que lee el mallado (vértices y polígonos de los triángulos)
void read_mesh(const std::string &filename, float **&vertex, uint32_t **&polygons, uint32_t &len_vertex, uint32_t &len_polygons) {

	uint32_t option = 2;

	if(option == 1){
		std::ifstream file(filename, std::ios::in | std::ios::binary);

		if(file.is_open()) {

			file.seekg(17);

			uint32_t dim;

			file.read ((char *) & dim, sizeof(uint32_t)); // dimensiones

			file.seekg(29);

			file.read ((char *) & len_vertex, sizeof(uint32_t)); // largo de vértices

			vertex = new float*[len_vertex];

			for(uint32_t i=0; i<len_vertex; i++) {
				vertex[i] = new float[3];

				for(uint32_t j=0; j<3; j++)
					file.read ((char *) & vertex[i][j], sizeof(float)); // lee cada punto (x,y,z)
			}

			file.seekg(4*3*len_vertex + 41);

			file.read ((char*)&len_polygons, sizeof(uint32_t)); // largo de polígonos
			polygons = new uint32_t*[len_polygons];

			for(uint32_t i = 0; i < len_polygons; i++) {
				polygons[i] = new uint32_t[3];

				for(uint32_t j = 0; j < 3; j++) 
					file.read ((char*)&polygons[i][j], sizeof(uint32_t)); // lee cada índice del triángulo (a,b,c)
			}
		}

		file.close();
	}

	else if(option == 2){

		std::vector<std::vector<float>> vertex_vector;
		std::vector<std::vector<int>> polygons_vector;
		int dim = 3;
		len_vertex = 0;		//Va aumentando mientras se van leyendo las líneas
		len_polygons = 0; 	//Idem
		
		std::ifstream file(filename);
		if(file.is_open()) {

			std::string id;
			std::string s1,s2,s3;
			std::string line;
			while(getline(file, line)){  
			    std::stringstream iss1(line);	//stringstream permite guardar strings y añadir o leerlos.
			    iss1>>id;	//le da el string de line a id
			    if (id=="v"){	//Para vértices, obtiene las coordenadas de todos los vertices y luego pasa a los poligonos.
			    	std::vector<float> vertex_coord(dim);
				    iss1>>vertex_coord[0];
				    iss1>>vertex_coord[1];
				    iss1>>vertex_coord[2];
				    len_vertex++;
				    vertex_vector.push_back(vertex_coord);
				    id = "n";
				} else if (id=="f"){
					std::vector<int> poly_vertex(dim);

				    iss1>>s1;
				    std::size_t pos = s1.find("//");	// encuentra la posición de "//" (contado caracter a caracter)... so, los poly vertex están separados por "//"					
					if(pos < 1000){
						poly_vertex[0] = std::stoi(s1.substr (pos+2))-1;
					}
					else{
						poly_vertex[0] = std::stoi(s1.substr(pos+1))-1;
					}

				    iss1>>s2;
					pos = s2.find("//"); 
					if(pos < 1000){
						poly_vertex[1] = std::stoi(s2.substr (pos+2))-1; 
					}
					else{
						poly_vertex[1] = std::stoi(s2.substr(pos+1))-1;
					}   

				    iss1>>s3;
					pos = s3.find("//");
					if(pos < 1000){
						poly_vertex[2] = std::stoi(s3.substr (pos+2))-1; 
					}
					else{
						poly_vertex[2] = std::stoi(s3.substr (pos+1))-1;
					}    
   
				    len_polygons++;
				    polygons_vector.push_back(poly_vertex);
				    id = "n";
				}

			}
		}

		file.close();

		vertex = new float*[len_vertex]; 
		for(uint32_t i=0; i<len_vertex; i++) {
			vertex[i] = new float[dim]; 
			vertex[i][0] = vertex_vector[i][0]; 
			vertex[i][1] = vertex_vector[i][1];
			vertex[i][2] = vertex_vector[i][2];
		}
		polygons = new uint32_t*[len_polygons];	
		for(uint32_t i=0; i<len_polygons; i++) {
			polygons[i] = new uint32_t[dim]; 	
			polygons[i][0] = polygons_vector[i][0]; 
			polygons[i][1] = polygons_vector[i][1];
			polygons[i][2] = polygons_vector[i][2];
			
			//std::cout << "Poligono i: " << polygons[i][0] << " " << polygons[i][1] << " " << polygons[i][2] << std::endl;
		}
	}
}

// función que lee un archivo .bundles
void read_bundles(const std::string &path, uint32_t &nBundles, uint32_t *&nFibers, uint32_t **&nPoints, float ****&Points) {

	std::vector<std::string> bundlesDir;

	DIR *dir;
	struct dirent *ent;
	if((dir = opendir(path.c_str())) != NULL) {
	  /* get all the file names and directories within directory */
	  while((ent = readdir (dir)) != NULL) {
	    const std::string bundle_dir = ent->d_name;
	    if(*bundle_dir.rbegin() == 'a')
	    	bundlesDir.emplace_back(path + bundle_dir);
	  }
	  closedir (dir);
	}
	else {
	  /* could not open directory */
	  perror("Error");
	  exit( EXIT_FAILURE );
	}

	std::sort(bundlesDir.begin(), bundlesDir.end());

	nBundles = bundlesDir.size();
	nFibers = new uint32_t[nBundles];
	nPoints = new uint32_t*[nBundles];
	Points = new float***[nBundles];

	for(uint32_t i=0; i<nBundles; i++) {

		std::string filename = bundlesDir[i];
		std::ifstream file(filename, std::ios::in | std::ios::binary);

		if(file.is_open()) {

			std::streampos fsize = 0;
			file.seekg( 0, std::ios::end );
			uint32_t num = (file.tellg() - fsize) / 4;
			file.seekg( 0 );

			uint32_t num_count = 0;

			std::vector<uint32_t> nPointsList;
			std::vector<float**> PointsList;

			while( num_count < num ) {

				uint32_t total_points; // number of points of each fiber
				file.read ((char*)&total_points, sizeof(uint32_t));

				float **Fiber = new float*[total_points]; // fiber

				for(uint32_t k=0; k<total_points; k++) {
					Fiber[k] = new float[3];

					for(uint32_t m = 0; m < 3; m++)
						file.read ((char *) & Fiber[k][m], sizeof(float));
				}

				num_count = num_count + 1 + ( total_points * 3 );
				nPointsList.emplace_back(total_points);
				PointsList.emplace_back(Fiber);
			}

			nFibers[i] = nPointsList.size();
			nPoints[i] = new uint32_t[nPointsList.size()];
			Points[i] = new float**[nPointsList.size()];

			for(uint32_t j=0; j<nPointsList.size(); j++) {
				nPoints[i][j] = nPointsList[j];
				Points[i][j] = PointsList[j];
			}
		}
		file.close();
	}
}

void write_intersection(const std::string &path, const std::string &pathBundles,
	const std::vector<std::vector<uint32_t>> &InTri, const std::vector<std::vector<uint32_t>> &FnTri,
	const std::vector<std::vector<std::vector<float>>> &InPoints, const std::vector<std::vector<std::vector<float>>> &FnPoints,
	const std::vector<std::vector<uint32_t>> &fib_index, const std::vector<std::vector<uint32_t>> &fiber_class) {
	
	std::vector<std::string> bundlesDir;

	DIR *dirInt;
	struct dirent *ent;
	if((dirInt = opendir(pathBundles.c_str())) != NULL) {
	  /* get all the file names and directories within directory */
	  while((ent = readdir (dirInt)) != NULL) {
	    const std::string bundle_dir = ent->d_name;
	    if(*bundle_dir.rbegin() == 's')
	    	bundlesDir.emplace_back(bundle_dir.substr(0,bundle_dir.size() - 8));
	  }
	  closedir (dirInt);
	}
	else {
	  /* could not open directory */
	  perror("Error");
	  exit( EXIT_FAILURE );
	}

	std::sort(bundlesDir.begin(), bundlesDir.end());

	// ============================================================================

	DIR *dir;
	if((dir = opendir(path.c_str())) == NULL) { // Checks if a directory path exists

		const int dir_err = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if(dir_err == -1) {
			perror("Error creating directory!");
			exit( EXIT_FAILURE );
		}
	}
	closedir (dir);

	for(uint32_t i=0; i<fib_index.size(); i++) {

		if(fib_index[i].size() == 0)
			continue;

		std::ofstream file(path + "/" + bundlesDir[i] + ".intersectiondata", std::ios::out | std::ios::binary);
		
		if(file.is_open()) {

			const uint32_t len_fibInd = fib_index[i].size();
			file.write(reinterpret_cast<const char*>( &len_fibInd ), sizeof( uint32_t )); // number of triangles

			for(uint32_t j=0; j<InTri[i].size(); j++) // writes index of each initial triangle
				file.write(reinterpret_cast<const char*>( &InTri[i][j] ), sizeof( uint32_t ));

			for(uint32_t j=0; j<FnTri[i].size(); j++) // writes index of each final triangle
				file.write(reinterpret_cast<const char*>( &FnTri[i][j] ), sizeof( uint32_t ));

			for(uint32_t j=0; j<InPoints[i].size(); j++) { // writes the initial intersection point (x,y,z)
				for(uint32_t k=0; k<3; k++) {
					file.write(reinterpret_cast<const char*>( &InPoints[i][j][k] ), sizeof( float ));
				}
			}

			for(uint32_t j=0; j<FnPoints[i].size(); j++) { // writes the final intersection point (x,y,z)
				for(uint32_t k=0; k<3; k++) {
					file.write(reinterpret_cast<const char*>( &FnPoints[i][j][k] ), sizeof( float ));
				}
			}

			for(uint32_t j=0; j<len_fibInd; j++){ // writes index of each fiber
				//std::cout << "Escribiendo fib_index: " << fib_index[i][j] << std::endl;
				file.write(reinterpret_cast<const char*>( &fib_index[i][j] ), sizeof( uint32_t ));
				
			}

			for(uint32_t j=0; j<len_fibInd; j++) // writes CLASS of each fiber
				file.write(reinterpret_cast<const char*>( &fiber_class[i][j] ), sizeof( uint32_t ));
		}
		file.close();
	}
}

void Delete(float **&vertex, uint32_t **&polygons, const uint32_t &len_vertex, const uint32_t &len_polygons) {

	for(uint32_t i=0; i<len_vertex; i++) {delete[] vertex[i]; delete[] vertex;}
	for(uint32_t i=0; i<len_polygons; i++) {delete[] polygons[i]; delete[] polygons;}
}

void write_membership(const std::string &path, const std::string &pathBundles, const std::vector<std::vector<uint32_t>> &hemi_membership){

	std::vector<std::string> bundlesDir;

	DIR *dirInt;
	struct dirent *ent;
	if((dirInt = opendir(pathBundles.c_str())) != NULL) {
	  /* get all the file names and directories within directory */
	  while((ent = readdir (dirInt)) != NULL) {
	    const std::string bundle_dir = ent->d_name;
	    if(*bundle_dir.rbegin() == 's')
	    	bundlesDir.emplace_back(bundle_dir.substr(0,bundle_dir.size() - 8));
	  }
	  closedir (dirInt);
	}
	else {
	  /* could not open directory */
	  perror("Error");
	  exit( EXIT_FAILURE );
	}

	std::sort(bundlesDir.begin(), bundlesDir.end());

	// ============================================================================

	DIR *dir;
	if((dir = opendir(path.c_str())) == NULL) { // Checks if a directory path exists

		const int dir_err = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if(dir_err == -1) {
			perror("Error creating directory!");
			exit( EXIT_FAILURE );
		}
	}
	closedir (dir);

	for(uint32_t i=0; i<hemi_membership.size(); i++) {

		std::ofstream file(path + "/" + bundlesDir[i] + ".membershipdata", std::ios::out | std::ios::binary);
		
		if(file.is_open()) {

			//std::cout << bundlesDir[i] <<  ":(" << hemi_membership[i][0] << "," << hemi_membership[i][1] << ")" << std::endl;

			for(uint32_t j=0; j<2; j++) // writes index of each initial triangle
				file.write(reinterpret_cast<const char*>( &hemi_membership[i][j] ), sizeof( uint8_t ));
		}
		file.close();
	}
}