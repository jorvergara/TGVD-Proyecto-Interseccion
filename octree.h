#ifndef OCTREE_H
#define OCTREE_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>

struct Triangle {
    std::array<std::array<double, 3>, 3> vertices;  // Coordenadas de los vértices (3x3)
    std::array<int, 3> indices;                    // Índices de los vértices
    std::array<double, 3> normal;                  // Normal del triángulo
    double area;                                   // Área del triángulo
    int idx;                                       // Índice del triángulo

    Triangle(const std::array<std::array<double, 3>, 3>& vertices, const std::array<int, 3>& indices, int idx);

    void calculate_normal();
    void calculate_area();
};

class OctreeNode {
public:
    std::array<double, 6> bounds;                 // [xmin, ymin, zmin, xmax, ymax, zmax]
    std::vector<Triangle> triangles;              // Triángulos en este nodo
    std::vector<OctreeNode> children;             // Hijos del nodo
    int depth;                                    // Profundidad del nodo
    int max_depth;                                // Profundidad máxima del octree
    int min_depth;                                // Profundidad mínima del octree
    double epsilon;                               // Tolerancia para subdivisión

    OctreeNode(const std::array<double, 6>& bounds, int depth = 0, int min_depth = 5, int max_depth = 7, double epsilon = 0.01);

    void insert_scene(const std::vector<Triangle>& scene);
    void subdivide();
    bool calculate_normal_variance();
    void recursive_subdivision();
};

bool is_triangle_in_bounds(const std::array<std::array<double, 3>, 3>& vertices, const std::array<double, 6>& bounds);

#endif
