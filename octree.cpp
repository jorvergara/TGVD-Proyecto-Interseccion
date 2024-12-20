#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 
#include <pybind11/stl.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

bool separating_axis_theorem(const std::array<std::array<float, 3>, 3>& triangle,
                             const std::array<float, 6>& bbox);
bool is_triangle_in_bounds(const std::array<std::array<float, 3>, 3>& triangle_vertices, 
                           const std::array<float, 6>& bounds);                   
bool moller_trumbore(
    const std::array<float, 3>& ray_origin, const std::array<float, 3>& ray_dir,
    const std::array<std::array<float, 3>, 3>& triangle
);

std::vector<std::vector<int>> ray_triangle_intersection(
    py::array_t<float> origins,
    py::array_t<float> directions,
    py::array_t<float> vertices,
    py::array_t<int> faces
);


int max_k_in_fiber_proyection(py::array_t<float> fiber_projection);
py::array_t<float> generate_points_interpolated(
    const std::array<float, 3>& p1, 
    const std::array<float, 3>& p2, 
    int k);

py::array_t<float> proyection_fiber(py::array_t<float> fiber, float tolerance, float vox_resolution);

class OctreeNode;  // Declaración adelantada

std::vector<std::vector<int>> intersection_octree(
    OctreeNode& octree,
    py::array_t<float> origins,
    py::array_t<float> directions,
    py::array_t<float> vertices,
    py::array_t<int> faces
);

class Triangle {
public:
    py::array_t<float> vertices;  // Coordenadas de los vértices como numpy array 3x3
    py::array_t<int> indices;     // Índices de los vértices como numpy array 1D
    py::array_t<float> normal;    // Normal del triángulo como numpy array 1D
    float area;                   // Área del triángulo
    int idx;                      // Índice del triángulo

    Triangle(py::array_t<float> vertices, py::array_t<int> indices, int idx)
        : vertices(std::move(vertices)),
          indices(std::move(indices)),
          idx(idx),
          normal(py::array_t<float>({3})) {
        calculate_normal();
        calculate_area();
    }

    py::array_t<float> get_vertices() const { return vertices; }
    py::array_t<int> get_indices() const { return indices; }
    py::array_t<float> get_normal() const { return normal; }
    float get_area() const { return area; }

private:
    void calculate_normal() {
        auto verts = vertices.unchecked<2>();
        auto norm = normal.mutable_unchecked<1>();

        const auto A = std::array<float, 3>{verts(0, 0), verts(0, 1), verts(0, 2)};
        const auto B = std::array<float, 3>{verts(1, 0), verts(1, 1), verts(1, 2)};
        const auto C = std::array<float, 3>{verts(2, 0), verts(2, 1), verts(2, 2)};

        std::array<float, 3> u = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
        std::array<float, 3> v = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};

        norm(0) = u[1] * v[2] - u[2] * v[1];
        norm(1) = u[2] * v[0] - u[0] * v[2];
        norm(2) = u[0] * v[1] - u[1] * v[0];

        float magnitude = std::sqrt(norm(0) * norm(0) + norm(1) * norm(1) + norm(2) * norm(2));
        if (magnitude > 0) {
            norm(0) /= magnitude;
            norm(1) /= magnitude;
            norm(2) /= magnitude;
        } else {
            norm(0) = norm(1) = norm(2) = 0.0f;
        }
    }

    void calculate_area() {
        auto verts = vertices.unchecked<2>();

        const auto A = std::array<float, 3>{verts(0, 0), verts(0, 1), verts(0, 2)};
        const auto B = std::array<float, 3>{verts(1, 0), verts(1, 1), verts(1, 2)};
        const auto C = std::array<float, 3>{verts(2, 0), verts(2, 1), verts(2, 2)};

        std::array<float, 3> u = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
        std::array<float, 3> v = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};

        std::array<float, 3> cross = {u[1] * v[2] - u[2] * v[1],
                                      u[2] * v[0] - u[0] * v[2],
                                      u[0] * v[1] - u[1] * v[0]};

        area = 0.5f * std::sqrt(std::inner_product(cross.begin(), cross.end(), cross.begin(), 0.0f));
    }
};

class OctreeNode {
public:
    py::array_t<float> bounds;         // [xmin, ymin, zmin, xmax, ymax, zmax]
    std::vector<Triangle> triangles;   // Lista de objetos Triangle
    std::vector<OctreeNode> children;  // Hijos del nodo
    int depth;                         // Profundidad del nodo
    int max_depth;                     // Profundidad máxima
    int min_depth;                     // Profundidad mínima 
    float epsilon;                     // Tolerancia para subdivisión

    OctreeNode(py::array_t<float> bounds, int depth = 0, int min_depth = 5, int max_depth = 7, float epsilon = 0.01f)
        : bounds(std::move(bounds)),
          depth(depth),
          min_depth(min_depth),
          max_depth(max_depth),
          epsilon(epsilon) {
        if (this->bounds.size() != 6) {
            throw std::invalid_argument("bounds debe ser un numpy array con 6 elementos: [xmin, ymin, zmin, xmax, ymax, zmax]");
        }
    }

    void insert_scene(py::array_t<float> scene_vertices, py::array_t<int> scene_indices) {
        // Validar forma de los vértices
        if (scene_vertices.ndim() != 2 || scene_vertices.shape(1) != 3) {
            throw std::invalid_argument("scene_vertices debe tener forma (n, 3): n vértices en 3D.");
        }

        // Validar forma de los índices
        if (scene_indices.ndim() != 2 || scene_indices.shape(1) != 3) {
            throw std::invalid_argument("scene_indices debe tener forma (m, 3): m triángulos definidos por 3 índices.");
        }

        size_t num_vertices = scene_vertices.shape(0);
        size_t num_triangles = scene_indices.shape(0);

        // Acceso sin comprobaciones para optimización
        auto vertices = scene_vertices.unchecked<2>();
        auto indices = scene_indices.unchecked<2>();

        for (size_t i = 0; i < num_triangles; ++i) {
            // Validar que los índices estén dentro del rango de los vértices
            for (int j = 0; j < 3; ++j) {
                if (indices(i, j) < 0 || indices(i, j) >= static_cast<int>(num_vertices)) {
                    throw std::out_of_range("Índice fuera del rango de los vértices.");
                }
            }

            // Crear el triángulo usando los índices
            py::array_t<float> tri_vertices({3, 3});
            auto tri_data = tri_vertices.mutable_unchecked<2>();

            for (int j = 0; j < 3; ++j) {
                int idx = indices(i, j);
                tri_data(j, 0) = vertices(idx, 0);
                tri_data(j, 1) = vertices(idx, 1);
                tri_data(j, 2) = vertices(idx, 2);
            }

            py::array_t<int> tri_indices({3});
            auto tri_indices_data = tri_indices.mutable_unchecked<1>();
            for (int j = 0; j < 3; ++j) {
                tri_indices_data(j) = indices(i, j);
            }

            // Agregar el triángulo a la lista
            triangles.emplace_back(tri_vertices, tri_indices, static_cast<int>(i));
        }
    }

    void subdivide() {
        auto b = bounds.unchecked<1>(); // Acceso rápido a los datos de los límites
        float xmin = b(0), ymin = b(1), zmin = b(2), xmax = b(3), ymax = b(4), zmax = b(5);
        float xmid = (xmin + xmax) / 2.0f, ymid = (ymin + ymax) / 2.0f, zmid = (zmin + zmax) / 2.0f;

        const std::array<std::array<float, 6>, 8> child_bounds = {{
        {xmin, ymin, zmin, xmid, ymid, zmid},
        {xmid, ymin, zmin, xmax, ymid, zmid},
        {xmin, ymid, zmin, xmid, ymax, zmid},
        {xmid, ymid, zmin, xmax, ymax, zmid},
        {xmin, ymin, zmid, xmid, ymid, zmax},
        {xmid, ymin, zmid, xmax, ymid, zmax},
        {xmin, ymid, zmid, xmid, ymax, zmax},
        {xmid, ymid, zmid, xmax, ymax, zmax}
        }};


        // Agregar cada hijo a la lista

        for (const auto& child : child_bounds) {
            py::array_t<float> child_bounds_array({6});
            auto child_data = child_bounds_array.mutable_unchecked<1>();

            for (size_t i = 0; i < 6; ++i) {
                child_data(i) = child[i];
            }

            children.emplace_back(OctreeNode(child_bounds_array, depth + 1, min_depth, max_depth, epsilon));
        }
    }

    bool calculate_normal_variance() const {
        // Verificar si no hay triángulos en el nodo
        if (triangles.empty()) {
            return false; // No subdividir
        }

        // Almacenar normales y áreas
        std::vector<std::array<float, 3>> normals;
        std::vector<float> areas;

        for (const auto& triangle : triangles) {
            auto normal = triangle.get_normal().unchecked<1>();
            float area = triangle.get_area();

            normals.push_back({normal(0), normal(1), normal(2)});
            areas.push_back(area);
        }

        // Crear numpy arrays para normales y áreas
        py::ssize_t num_triangles = static_cast<py::ssize_t>(normals.size());
        std::vector<py::ssize_t> normals_dims = {num_triangles, 3};
        std::vector<py::ssize_t> areas_dims = {num_triangles};

        py::array_t<float> normals_array(normals_dims);
        py::array_t<float> areas_array(areas_dims);

        auto normals_mutable = normals_array.mutable_unchecked<2>();
        auto areas_mutable = areas_array.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < num_triangles; ++i) {
            for (py::ssize_t j = 0; j < 3; ++j) {
                normals_mutable(i, j) = normals[i][j];
            }
            areas_mutable(i) = areas[i];
        }

        // Calcular el promedio ponderado de las normales
        std::array<float, 3> avg_normal = {0.0f, 0.0f, 0.0f};
        float total_area = 0.0f;

        for (py::ssize_t i = 0; i < num_triangles; ++i) {
            total_area += areas_mutable(i);
            for (py::ssize_t j = 0; j < 3; ++j) {
                avg_normal[j] += normals_mutable(i, j) * areas_mutable(i);
            }
        }

        for (py::ssize_t j = 0; j < 3; ++j) {
            avg_normal[j] /= total_area; // Promedio ponderado
        }

        // Normalizar el promedio
        float magnitude = std::sqrt(avg_normal[0] * avg_normal[0] +
                                    avg_normal[1] * avg_normal[1] +
                                    avg_normal[2] * avg_normal[2]);

        if (magnitude > 0) {
            for (py::ssize_t j = 0; j < 3; ++j) {
                avg_normal[j] /= magnitude;
            }
        }

        // Calcular L_r^{2,1} (varianza)
        float variance = 0.0f;
        for (py::ssize_t i = 0; i < num_triangles; ++i) {
            float diff[3] = {
                normals_mutable(i, 0) - avg_normal[0],
                normals_mutable(i, 1) - avg_normal[1],
                normals_mutable(i, 2) - avg_normal[2]
            };

            float norm_diff = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
            variance += norm_diff * norm_diff;
        }

        // Retornar si la varianza excede el epsilon
        return variance > epsilon;
    }

    void recursive_subdivision() {
        if (triangles.size() <= 1) {
            return;  // No subdividir si la lista tiene 1 o menos triángulos
        }

        if (depth >= max_depth) {
            return;  // Se alcanzó la profundidad máxima
        }

        if (depth >= min_depth && !calculate_normal_variance()) {
            return;  // La variación de normales es menor que la tolerancia
        }

        // Subdividir el nodo actual
        subdivide();

        // Redistribuir los triángulos
        redistribute_triangles();
        
        // Llamada recursiva para cada hijo
        for (auto& child : children) {
            child.recursive_subdivision();
        }
    }

    void recursive_subdivision_vertex() {
        if (triangles.size() <= 1) {
            return;  // No subdividir si la lista tiene 1 o menos triángulos
        }

        if (depth >= max_depth) {
            return;  // Se alcanzó la profundidad máxima
        }

        if (depth >= min_depth && !calculate_normal_variance()) {
            return;  // La variación de normales es menor que la tolerancia
        }

        // Subdividir el nodo actual
        subdivide();

        // Redistribuir los triángulos
        redistribute_triangles_vertex();
        
        // Llamada recursiva para cada hijo
        for (auto& child : children) {
            child.recursive_subdivision_vertex();
        }
    }

    
    
    void redistribute_triangles() {
        // Redistribuir triángulos entre los hijos
        
        for (const auto& triangle : triangles) {
            // Convertir `triangle.vertices` a std::array<std::array<float, 3>, 3>
            auto vertices = triangle.get_vertices().unchecked<2>();
            std::array<std::array<float, 3>, 3> tri_vertices = {{
                {vertices(0, 0), vertices(0, 1), vertices(0, 2)},
                {vertices(1, 0), vertices(1, 1), vertices(1, 2)},
                {vertices(2, 0), vertices(2, 1), vertices(2, 2)}
            }};
            for (auto& child : children) {
                
                // Convertir `child.bounds` a std::array<float, 6>
                auto bbox = child.bounds.unchecked<1>();

                // Verificar intersección usando SAT
                if (separating_axis_theorem(tri_vertices, {bbox(0), bbox(1), bbox(2), bbox(3), bbox(4), bbox(5)})) {
                    child.triangles.push_back(triangle);
                }
            }
        }
        // Limpiar los triángulos del nodo actual
        triangles.clear();
    }

    void redistribute_triangles_vertex() {
        // Redistribuir triángulos entre los hijos
        
        for (const auto& triangle : triangles) {
            // Convertir `triangle.vertices` a std::array<std::array<float, 3>, 3>
            auto vertices = triangle.get_vertices().unchecked<2>();
            std::array<std::array<float, 3>, 3> tri_vertices = {{
                {vertices(0, 0), vertices(0, 1), vertices(0, 2)},
                {vertices(1, 0), vertices(1, 1), vertices(1, 2)},
                {vertices(2, 0), vertices(2, 1), vertices(2, 2)}
            }};
            for (auto& child : children) {
                
                // Convertir `child.bounds` a std::array<float, 6>
                auto bbox = child.bounds.unchecked<1>();

                // Verificar intersección usando SAT
                if (is_triangle_in_bounds(tri_vertices, {bbox(0), bbox(1), bbox(2), bbox(3), bbox(4), bbox(5)})) {
                    child.triangles.push_back(triangle);
                }
            }
        }
        // Limpiar los triángulos del nodo actual
        // triangles.clear();
    }

    OctreeNode* find_leaf_containing_point(const std::array<float, 3>& point) {
        // Acceso sin comprobaciones para optimización
        auto b = bounds.unchecked<1>();
        float xmin = b(0), ymin = b(1), zmin = b(2);
        float xmax = b(3), ymax = b(4), zmax = b(5);

        // Verificar si el punto está dentro de los límites del nodo actual
        if (!(xmin <= point[0] && point[0] <= xmax &&
              ymin <= point[1] && point[1] <= ymax &&
              zmin <= point[2] && point[2] <= zmax)) {
            return nullptr;  // El punto está fuera de los límites
        }

        // Si no tiene hijos, este es un nodo hoja
        if (children.empty()) {
            return this;
        }

        // Buscar en los hijos
        for (auto& child : children) {
            OctreeNode* result = child.find_leaf_containing_point(point);
            if (result != nullptr) {
                return result;
            }
        }
        return nullptr;  // No se encontró en ninguno de los hijos
    }
    std::vector<int> get_adjacent_triangles(const std::array<float, 3>& point) {
        // Encontrar el nodo hoja que contiene el punto
        OctreeNode* leaf_node = find_leaf_containing_point(point);
        if (!leaf_node) {
            return {}; // Retornar vacío si no se encuentra el nodo
        }

        // Obtener los límites y el punto central del nodo
        auto bounds_data = leaf_node->bounds.unchecked<1>();
        float x0 = bounds_data(0), y0 = bounds_data(1), z0 = bounds_data(2);
        float x1 = bounds_data(3), y1 = bounds_data(4), z1 = bounds_data(5);
        std::array<float, 3> central_point = {
            (x0 + x1) / 2.0f,
            (y0 + y1) / 2.0f,
            (z0 + z1) / 2.0f
        };

        // Calcular los puntos adyacentes
        std::vector<std::array<float, 3>> adjacent_points;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue; // Excluir el punto central
                    adjacent_points.push_back({
                        central_point[0] + dx * (x1 - x0),
                        central_point[1] + dy * (y1 - y0),
                        central_point[2] + dz * (z1 - z0)
                    });
                }
            }
        }

        // Encontrar los nodos adyacentes
        std::unordered_set<OctreeNode*> adjacent_nodes;
        for (const auto& pt : adjacent_points) {
            OctreeNode* adjacent_node = find_leaf_containing_point(pt);
            if (adjacent_node) {
                adjacent_nodes.insert(adjacent_node);
            }
        }
        adjacent_nodes.insert(leaf_node); // Incluir el nodo actual

        // Recopilar triángulos únicos
        std::unordered_set<int> unique_triangles;
        for (OctreeNode* node : adjacent_nodes) {
            for (const auto& triangle : node->triangles) {
                unique_triangles.insert(triangle.idx);
            }
        }

        // Convertir a vector y retornar
        return std::vector<int>(unique_triangles.begin(), unique_triangles.end());
    }

    std::vector<OctreeNode*> get_leaf_nodes() {
        std::vector<OctreeNode*> leaf_nodes;
        if (children.empty()) {
            leaf_nodes.push_back(this);
            return leaf_nodes;
        }
        for (auto& child : children) {
            auto child_leaf_nodes = child.get_leaf_nodes();
            leaf_nodes.insert(leaf_nodes.end(), child_leaf_nodes.begin(), child_leaf_nodes.end());
        }
        return leaf_nodes;
    }

    py::array_t<int> fill_lookup_table() {
        auto b = bounds.unchecked<1>();
        int max_subdiv = 1 << max_depth;

        // Crear matriz de búsqueda
        py::array_t<int> lookup_table({max_subdiv, max_subdiv, max_subdiv});
        auto table = lookup_table.mutable_unchecked<3>();

        float x_range = (b(3) - b(0)) / max_subdiv;
        float y_range = (b(4) - b(1)) / max_subdiv;
        float z_range = (b(5) - b(2)) / max_subdiv;

        auto leaf_nodes = get_leaf_nodes();

        // Llenar tabla de búsqueda
        #pragma omp parallel for
        for (size_t idx = 0; idx < leaf_nodes.size(); ++idx) {
            auto& leaf = *leaf_nodes[idx];
            auto lb = leaf.bounds.unchecked<1>();

            int x_start = static_cast<int>(std::round((lb(0) - b(0)) / x_range));
            int y_start = static_cast<int>(std::round((lb(1) - b(1)) / y_range));
            int z_start = static_cast<int>(std::round((lb(2) - b(2)) / z_range));

            int x_end = static_cast<int>(std::round((lb(3) - b(0)) / x_range));
            int y_end = static_cast<int>(std::round((lb(4) - b(1)) / y_range));
            int z_end = static_cast<int>(std::round((lb(5) - b(2)) / z_range));

            for (int x = x_start; x < x_end; ++x) {
                for (int y = y_start; y < y_end; ++y) {
                    for (int z = z_start; z < z_end; ++z) {
                        table(x, y, z) = idx;
                    }
                }
            }
        }
        return lookup_table;
    }
    py::array_t<float> calculate_xyz_range() {
        auto b = bounds.unchecked<1>();
        py::array_t<float> result(3);
        auto result_mut = result.mutable_unchecked<1>();
        result_mut(0) = (b(3) - b(0)) / std::pow(2.0f, max_depth);
        result_mut(1) = (b(4) - b(1)) / std::pow(2.0f, max_depth);
        result_mut(2) = (b(5) - b(2)) / std::pow(2.0f, max_depth);
        return result;
    }

    py::array_t<float> interpolate_fiber(py::array_t<float> fiber) {
        auto xyz_range = calculate_xyz_range().unchecked<1>();
        auto f = fiber.unchecked<2>();
        int n_points = f.shape(0);

        // Declara el array de tipo float con dimensiones (n_points, 3)
        py::array_t<float> fiber_i({n_points, 3});

        auto fi_mut = fiber_i.mutable_unchecked<2>();
        for (size_t i = 0; i < n_points; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                fi_mut(i, j) = f(i, j) / xyz_range(j);
            }
        }

        auto fiber_i_x = fiber_i.attr("__getitem__")(py::make_tuple(py::slice(0, n_points, 1), 0));
        auto fiber_i_y = fiber_i.attr("__getitem__")(py::make_tuple(py::slice(0, n_points, 1), 1));
        auto fiber_i_z = fiber_i.attr("__getitem__")(py::make_tuple(py::slice(0, n_points, 1), 2));

        int k = std::max({
            max_k_in_fiber_proyection(fiber_i_x),
            max_k_in_fiber_proyection(fiber_i_y),
            max_k_in_fiber_proyection(fiber_i_z)
        });

        
        std::vector<std::array<float, 3>> interpolated_fiber;
        interpolated_fiber.push_back({f(0, 0), f(0, 1), f(0, 2)});

        for (size_t i = 0; i < n_points - 1; ++i) {
            auto p1 = std::array<float, 3>{f(i, 0), f(i, 1), f(i, 2)};
            auto p2 = std::array<float, 3>{f(i + 1, 0), f(i + 1, 1), f(i + 1, 2)};

            auto interp_points = generate_points_interpolated(p1, p2, k);

            for (ssize_t j = 1; j < interp_points.shape(0); ++j) {
                auto point = interp_points.mutable_unchecked<2>();
                std::array<float, 3> new_point = {point(j, 0), point(j, 1), point(j, 2)};
                interpolated_fiber.push_back(new_point);
            }
        }
        
        int n_interpolated_points = static_cast<int>(interpolated_fiber.size());
        py::array_t<float> result({n_interpolated_points, 3});
        auto result_mut = result.mutable_unchecked<2>();

        for (size_t i = 0; i < interpolated_fiber.size(); ++i) {
            for (size_t j = 0; j < 3; ++j) {
                result_mut(i, j) = interpolated_fiber[i][j];
            }
        }
        return result;
    }
};

bool is_triangle_in_bounds(const std::array<std::array<float, 3>, 3>& triangle_vertices, 
                           const std::array<float, 6>& bounds) {
    // Descomposición de los límites
    float xmin = bounds[0], ymin = bounds[1], zmin = bounds[2];
    float xmax = bounds[3], ymax = bounds[4], zmax = bounds[5];

    // Verificar si algún vértice está dentro de los límites
    for (const auto& vertex : triangle_vertices) {
        float vx = vertex[0], vy = vertex[1], vz = vertex[2];
        if (vx >= xmin && vx <= xmax &&
            vy >= ymin && vy <= ymax &&
            vz >= zmin && vz <= zmax) {
            return true;  // Al menos un vértice está dentro de los límites
        }
    }
    return false;  // Ningún vértice está dentro de los límites
}

bool separating_axis_theorem(const std::array<std::array<float, 3>, 3>& triangle,
                             const std::array<float, 6>& bbox) {
    // Descomponer entrada
    const auto& v0 = triangle[0];
    const auto& v1 = triangle[1];
    const auto& v2 = triangle[2];
    std::array<float, 3> bbox_min = {bbox[0], bbox[1], bbox[2]};
    std::array<float, 3> bbox_max = {bbox[3], bbox[4], bbox[5]};

    // Función auxiliar para proyecciones
    auto project = [](const std::vector<std::array<float, 3>>& points, const std::array<float, 3>& axis) {
        std::vector<float> projections;
        for (const auto& point : points) {
            projections.push_back(point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2]);
        }
        return projections;
    };

    // 1. Ejes de la bounding box
    std::array<std::array<float, 3>, 3> box_axes = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

    // 2. Ejes del triángulo
    std::array<std::array<float, 3>, 3> edges = {{
        {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]},
        {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]},
        {v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]}
    }};

    // Normal del triángulo
    std::array<float, 3> triangle_normal = {
        edges[0][1] * edges[1][2] - edges[0][2] * edges[1][1],
        edges[0][2] * edges[1][0] - edges[0][0] * edges[1][2],
        edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0]
    };

    // 3. Ejes de los productos cruzados (triángulo x caja)
    std::vector<std::array<float, 3>> cross_axes;
    for (const auto& edge : edges) {
        for (const auto& axis : box_axes) {
            cross_axes.push_back({
                edge[1] * axis[2] - edge[2] * axis[1],
                edge[2] * axis[0] - edge[0] * axis[2],
                edge[0] * axis[1] - edge[1] * axis[0]
            });
        }
    }

    // Combinar todos los ejes relevantes
    std::vector<std::array<float, 3>> axes(box_axes.begin(), box_axes.end());
    axes.push_back(triangle_normal);
    axes.insert(axes.end(), cross_axes.begin(), cross_axes.end());

    // Proyección de la caja en el eje
    std::vector<std::array<float, 3>> box_corners = {
        {bbox_min[0], bbox_min[1], bbox_min[2]},
        {bbox_min[0], bbox_min[1], bbox_max[2]},
        {bbox_min[0], bbox_max[1], bbox_min[2]},
        {bbox_min[0], bbox_max[1], bbox_max[2]},
        {bbox_max[0], bbox_min[1], bbox_min[2]},
        {bbox_max[0], bbox_min[1], bbox_max[2]},
        {bbox_max[0], bbox_max[1], bbox_min[2]},
        {bbox_max[0], bbox_max[1], bbox_max[2]}
    };
    // 4. Probar cada eje
    for (auto& axis : axes) {
        // Ignorar ejes degenerados
        float magnitude = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        if (magnitude < 1e-6) {
            continue;
        }

        // Normalizar eje
        axis = {axis[0] / magnitude, axis[1] / magnitude, axis[2] / magnitude};

        // Proyección del triángulo en el eje
        std::vector<std::array<float, 3>> triangle_points = {v0, v1, v2};
        auto triangle_proj = project(triangle_points, axis);
        float triangle_min = *std::min_element(triangle_proj.begin(), triangle_proj.end());
        float triangle_max = *std::max_element(triangle_proj.begin(), triangle_proj.end());

        auto box_proj = project(box_corners, axis);
        float box_min = *std::min_element(box_proj.begin(), box_proj.end());
        float box_max = *std::max_element(box_proj.begin(), box_proj.end());

        // Verificar si las proyecciones no se solapan
        if (triangle_max < box_min || box_max < triangle_min) {
            return false;  // Separación encontrada
        }
    }

    return true;  // No hay separación, las geometrías se intersectan
}

// Resta de vectores
void vector_subtract(const std::array<float, 3>& a, const std::array<float, 3>& b, std::array<float, 3>& result) {
    for (size_t i = 0; i < 3; ++i) {
        result[i] = a[i] - b[i];
    }
}

// Producto cruzado
void vector_cross(const std::array<float, 3>& a, const std::array<float, 3>& b, std::array<float, 3>& result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// Producto punto
float vector_dot(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Longitud de un vector
float vector_length(const std::array<float, 3>& a) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

// Normalización de un vector
void vector_normalize(std::array<float, 3>& a) {
    float length = vector_length(a);
    if (length > 0.0f) {
        for (size_t i = 0; i < 3; ++i) {
            a[i] /= length;
        }
    }
}

bool moller_trumbore(
    const std::array<float, 3>& ray_origin, const std::array<float, 3>& ray_dir,
    const std::array<std::array<float, 3>, 3>& triangle) {

    const float EPSILON = 1e-6f;

    std::array<float, 3> normalized_ray_dir = ray_dir;
    float ray_length = vector_length(normalized_ray_dir);
    vector_normalize(normalized_ray_dir);

    // Extrae los vértices del triángulo
    const std::array<float, 3>& vertex0 = triangle[0];
    const std::array<float, 3>& vertex1 = triangle[1];
    const std::array<float, 3>& vertex2 = triangle[2];

    // Bordes del triángulo
    std::array<float, 3> edge1, edge2;
    vector_subtract(vertex1, vertex0, edge1);
    vector_subtract(vertex2, vertex0, edge2);

    // Calcula el determinante
    std::array<float, 3> h;
    vector_cross(normalized_ray_dir, edge2, h);
    float a = vector_dot(edge1, h);

    if (std::abs(a) < EPSILON) {
        return false; // El rayo es paralelo al triángulo
    }

    // Calcula el parámetro f
    float f = 1.0f / a;
    std::array<float, 3> s;
    vector_subtract(ray_origin, vertex0, s);
    float u = f * vector_dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false; // El punto está fuera del triángulo
    }

    std::array<float, 3> q;
    vector_cross(s, edge1, q);
    float v = f * vector_dot(normalized_ray_dir, q);

    if (v < 0.0f || u + v > 1.0f) {
        return false; // El punto está fuera del triángulo
    }

    // Calcula t, el parámetro de la ecuación del rayo
    float t = f * vector_dot(edge2, q);

    // El punto de intersección debe estar a lo largo del rayo
    return t > EPSILON && t <= ray_length;
}

std::vector<std::vector<int>> ray_triangle_intersection(
    py::array_t<float> origins,
    py::array_t<float> directions,
    py::array_t<float> vertices,
    py::array_t<int> faces
) {
    auto origins_unchecked = origins.unchecked<3>();     // (n_fibers, n_segments, 3)
    auto directions_unchecked = directions.unchecked<3>(); // (n_fibers, n_segments, 3)
    auto vertices_unchecked = vertices.unchecked<2>();   // (n_vertices, 3)
    auto faces_unchecked = faces.unchecked<2>();         // (n_faces, 3)

    size_t n_fibers = origins.shape(0);
    size_t n_segments = origins.shape(1);
    size_t n_faces = faces.shape(0);

    // Precalcular todos los triángulos
    std::vector<std::array<std::array<float, 3>, 3>> precomputed_triangles(n_faces);
    for (size_t f = 0; f < n_faces; ++f) {
        for (int k = 0; k < 3; ++k) {
            precomputed_triangles[f][k][0] = vertices_unchecked(faces_unchecked(f, k), 0);
            precomputed_triangles[f][k][1] = vertices_unchecked(faces_unchecked(f, k), 1);
            precomputed_triangles[f][k][2] = vertices_unchecked(faces_unchecked(f, k), 2);
        }
    }

    // Estructura para almacenar intersecciones
    std::vector<std::vector<int>> intersections(n_fibers);

    // Procesar todos los rayos de forma paralela
    #pragma omp parallel for
    for (size_t i = 0; i < n_fibers; ++i) {
        std::vector<int> local_intersections;
        for (size_t j = 0; j < n_segments; ++j) {
            // Convertir origen y dirección del rayo a std::array
            std::array<float, 3> ray_origin = {
                origins_unchecked(i, j, 0),
                origins_unchecked(i, j, 1),
                origins_unchecked(i, j, 2)
            };
            std::array<float, 3> ray_direction = {
                directions_unchecked(i, j, 0),
                directions_unchecked(i, j, 1),
                directions_unchecked(i, j, 2)
            };

            // Verificar intersección con cada triángulo
            for (size_t f = 0; f < n_faces; ++f) {
                if (moller_trumbore(ray_origin, ray_direction, precomputed_triangles[f])) {
                    local_intersections.push_back(f);
                }
            }
        }

        // Combinar los resultados locales en la estructura global
        #pragma omp critical
        intersections[i] = std::move(local_intersections);
    }
    return intersections;
}

std::vector<std::vector<int>> intersection_octree(
    OctreeNode& octree,
    py::array_t<float> origins,
    py::array_t<float> directions,
    py::array_t<float> vertices,
    py::array_t<int> faces) {
    
    auto origins_unchecked = origins.unchecked<3>();     // (n_fibers, n_segments, 3)
    auto directions_unchecked = directions.unchecked<3>(); // (n_fibers, n_segments, 3)
    auto vertices_unchecked = vertices.unchecked<2>();   // (n_vertices, 3)
    auto faces_unchecked = faces.unchecked<2>();         // (n_faces, 3)

    size_t n_fibers = origins.shape(0);
    size_t n_segments = origins.shape(1);

    // Estructura para almacenar las intersecciones
    std::vector<std::vector<int>> intersections(n_fibers);

    // Paralelizar las consultas por fibra
    #pragma omp parallel for
    for (size_t i = 0; i < n_fibers; ++i) {
        std::unordered_set<int> local_intersections;  // Para evitar duplicados

        for (size_t j = 0; j < n_segments; ++j) {
            // Convertir origen y dirección del rayo a std::array
            std::array<float, 3> ray_origin = {
                origins_unchecked(i, j, 0),
                origins_unchecked(i, j, 1),
                origins_unchecked(i, j, 2)
            };
            std::array<float, 3> ray_direction = {
                directions_unchecked(i, j, 0),
                directions_unchecked(i, j, 1),
                directions_unchecked(i, j, 2)
            };

            // Obtener triángulos adyacentes usando el Octree
            auto adjacent_triangles = octree.get_adjacent_triangles(ray_origin);

            // Verificar intersección con cada triángulo adyacente
            for (int triangle_idx : adjacent_triangles) {
                std::array<std::array<float, 3>, 3> triangle = {{
                    {vertices_unchecked(faces_unchecked(triangle_idx, 0), 0),
                     vertices_unchecked(faces_unchecked(triangle_idx, 0), 1),
                     vertices_unchecked(faces_unchecked(triangle_idx, 0), 2)},
                    {vertices_unchecked(faces_unchecked(triangle_idx, 1), 0),
                     vertices_unchecked(faces_unchecked(triangle_idx, 1), 1),
                     vertices_unchecked(faces_unchecked(triangle_idx, 1), 2)},
                    {vertices_unchecked(faces_unchecked(triangle_idx, 2), 0),
                     vertices_unchecked(faces_unchecked(triangle_idx, 2), 1),
                     vertices_unchecked(faces_unchecked(triangle_idx, 2), 2)}
                }};

                if (moller_trumbore(ray_origin, ray_direction, triangle)) {
                    local_intersections.insert(triangle_idx);
                }
            }
        }
        // Convertir a vector y almacenar en las intersecciones globales
        intersections[i] = std::vector<int>(local_intersections.begin(), local_intersections.end());
    }
    return intersections;
}

// Función para calcular k puntos entre dos puntos
int calculate_k_points(float p1, float p2, int w = 3, int delta = 1) {
    float d_max = w - delta;
    float d_min = (p1 < p2) ? std::floor(p2) - std::ceil(p1) : std::floor(p1) - std::ceil(p2);
    int k = static_cast<int>(std::ceil(d_min/d_max));
    if (k < 1) return 0;
    return k;
}

// Función para generar puntos interpolados en una dimensión
py::array_t<float> generate_points_interpolated(
    const std::array<float, 3>& p1, 
    const std::array<float, 3>& p2, 
    int k) {

    std::array<float, 3> distancia = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    float distance_norm = std::sqrt(distancia[0]*distancia[0] + 
                                    distancia[1]*distancia[1] + 
                                    distancia[2]*distancia[2]);

    // Si la distancia es cero, devolver solo p1 y p2
    if (distance_norm == 0) {
        py::array_t<float> result({2, 3});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < 3; ++i) {
            r(0, i) = p1[i];
            r(1, i) = p2[i];
        }
        return result;
    }

    int num_points = k + 2;
    py::array_t<float> result({num_points, 3});
    auto r = result.mutable_unchecked<2>();

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < 3; ++j) {
            r(i, j) = p1[j] + i * distancia[j] / (k + 1);
        }
    }

    return result;
}

// Función para calcular el máximo k en una proyección de fibra
int max_k_in_fiber_proyection(py::array_t<float> fiber_projection) {
    auto fp = fiber_projection.unchecked<1>();
    size_t n_points = fiber_projection.shape(0);

    if (n_points < 2) {
        throw std::invalid_argument("La proyección de la fibra debe tener al menos 2 puntos.");
    }

    size_t p1 = 0;
    float max_distance = 0.0f;

    for (size_t i = 0; i < n_points - 1; ++i) {
        float dist = std::abs(fp(i + 1) - fp(i));
        if (dist > max_distance) {
            max_distance = dist;
            p1 = i;
        }
    }

    size_t p2 = p1 + 1;
    int k = calculate_k_points(fp(p1), fp(p2));
    return k;
}

std::pair<py::array_t<float>, py::array_t<float>> compute_directions_fiber(py::array_t<float> fiber) {
    auto f = fiber.unchecked<2>();
    int n_points = fiber.shape(0);

    // Prealocar memoria
    py::array_t<float> origin({n_points - 1, 3});
    py::array_t<float> direction({n_points - 1, 3});

    auto o = origin.mutable_unchecked<2>();
    auto d = direction.mutable_unchecked<2>();

    // Calcular orígenes y direcciones
    for (size_t i = 0; i < n_points - 1; ++i) {
        // Punto de origen
        o(i, 0) = f(i, 0);
        o(i, 1) = f(i, 1);
        o(i, 2) = f(i, 2);

        // Dirección
        d(i, 0) = f(i + 1, 0) - f(i, 0);
        d(i, 1) = f(i + 1, 1) - f(i, 1);
        d(i, 2) = f(i + 1, 2) - f(i, 2);
    }
    return {origin, direction};
}


std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> find_intersections(
    OctreeNode& octree,
    py::array_t<float> centered_bundle,
    py::array_t<float> centered_vertices,  
    py::array_t<int> polygons_lh) {
    
    auto bundle = centered_bundle.unchecked<3>();
    auto vertices = centered_vertices.unchecked<2>();
    auto polygons = polygons_lh.unchecked<2>();

    int max_subdiv = 1 << octree.max_depth;
    auto xyz_range = octree.calculate_xyz_range().unchecked<1>();
    
    py::array_t<int> look_up_table = octree.fill_lookup_table();
    auto table = look_up_table.mutable_unchecked<3>();   

    std::vector<OctreeNode*> leaf_nodes = octree.get_leaf_nodes();

    std::vector<std::vector<int>> intersection(bundle.shape(0));
    std::vector<std::vector<int>> adjacent_triangles(bundle.shape(0));

    std::vector<std::vector<std::vector<std::vector<int>>>> adjacent_triangles_3d(max_subdiv,std::vector<std::vector<std::vector<int>>>(max_subdiv, std::vector<std::vector<int>>(max_subdiv)));
    std::vector<std::vector<std::vector<std::vector<int>>>> adjacent_nodes_3d(max_subdiv, std::vector<std::vector<std::vector<int>>>(max_subdiv, std::vector<std::vector<int>>(max_subdiv)));
    std::vector<std::vector<std::vector<bool>>> adjacent_triangles_flag(max_subdiv, std::vector<std::vector<bool>>(max_subdiv, std::vector<bool>(max_subdiv, false)));

    std::vector<int> dx = {-1, 0, 1};

    for (size_t fiber = 0; fiber < bundle.shape(0); ++fiber) {
        std::vector<int> intersection_per_fiber;
        std::vector<int> adjacent_triangles_per_fiber;
        
        auto projected_fiber = proyection_fiber(centered_bundle.attr("__getitem__")(fiber), 5.0, 0.7);
        auto fiber_i = octree.interpolate_fiber(projected_fiber);
        auto origins_directions = compute_directions_fiber(fiber_i);

        auto origin_i = origins_directions.first.unchecked<2>();
        auto direction_i = origins_directions.second.unchecked<2>();
        

        for (size_t point = 0; point < origin_i.shape(0); ++point) {
            std::array<float, 3> centered_point = {
                origin_i(point, 0),
                origin_i(point, 1),
                origin_i(point, 2)
            };
            std::array<float, 3> centered_direction = {
                direction_i(point, 0),
                direction_i(point, 1),
                direction_i(point, 2)                
            };
            int p_xi = std::floor(centered_point[0]/xyz_range(0));
            int p_yi = std::floor(centered_point[1]/xyz_range(1));
            int p_zi = std::floor(centered_point[2]/xyz_range(2));

            if (p_xi < 0 || p_yi < 0 || p_zi < 0 || p_xi >= max_subdiv || p_yi >= max_subdiv || p_zi >= max_subdiv) {
                continue;
            }

            std::vector<int> adjacent_triangles_i;
            std::vector<int> adjacent_nodes_i;

            if (adjacent_triangles_flag[p_xi][p_yi][p_zi]) {
                adjacent_nodes_i = adjacent_nodes_3d[p_xi][p_yi][p_zi];
                adjacent_triangles_i = adjacent_triangles_3d[p_xi][p_yi][p_zi];
            }
            else{
                std::vector<std::array<int, 3>> adjacent_idx;
                // Generar índices adyacentes
                for (int i : dx) {
                    for (int j : dx) {
                        for (int k : dx) {
                            int xi = p_xi + i;
                            int yi = p_yi + j;
                            int zi = p_zi + k;

                            if (xi >= 0 && yi >= 0 && zi >= 0 && xi < max_subdiv && yi < max_subdiv && zi < max_subdiv) {
                                adjacent_idx.push_back({xi, yi, zi});
                                
                            }
                        }
                    }
                }
                //Obtener nodos adyacentes
                std::set<int> nodes_set;
                for (const auto& idx : adjacent_idx) {
                    int node = table(idx[0], idx[1], idx[2]);
                    nodes_set.insert(node);
                }
                
                adjacent_nodes_i.assign(nodes_set.begin(), nodes_set.end());
                adjacent_nodes_3d[p_xi][p_yi][p_zi] = adjacent_nodes_i;

                //Obtener triángulos adyacentes
                std::set<int> unique_triangles;
                for (int node : adjacent_nodes_i) {
                    for (const auto& triangle : leaf_nodes[node]->triangles) {
                        unique_triangles.insert(triangle.idx);  // Evitar duplicados
                    }
                }
                // Convertir de set a vector
                adjacent_triangles_i.assign(unique_triangles.begin(), unique_triangles.end());
                adjacent_triangles_3d[p_xi][p_yi][p_zi] = adjacent_triangles_i;
                adjacent_triangles_flag[p_xi][p_yi][p_zi] = true;
            }
            if (adjacent_triangles_i.size() == 0) {
                continue;
            }
            // Extender triángulos adyacentes por fibra
            adjacent_triangles_per_fiber.insert(
                adjacent_triangles_per_fiber.end(), 
                adjacent_triangles_i.begin(), 
                adjacent_triangles_i.end()
            );
            
            // Comprobar intersección usando el método de Möller-Trumbore
            for (int idx : adjacent_triangles_i) {
                auto polygon_vertices = std::array<std::array<float, 3>, 3> {{
                    {vertices(polygons(idx, 0), 0), vertices(polygons(idx, 0), 1), vertices(polygons(idx, 0), 2)},
                    {vertices(polygons(idx, 1), 0), vertices(polygons(idx, 1), 1), vertices(polygons(idx, 1), 2)},
                    {vertices(polygons(idx, 2), 0), vertices(polygons(idx, 2), 1), vertices(polygons(idx, 2), 2)}
                }};
                if (moller_trumbore(centered_point, centered_direction, polygon_vertices)) {
                    intersection_per_fiber.push_back(idx);
                }
            }
        }
        // Eliminar duplicados usando std::set y asignar a los vectores
        std::set<int> unique_intersection_per_fiber(intersection_per_fiber.begin(), intersection_per_fiber.end());
        std::set<int> unique_adjacent_triangles_per_fiber(adjacent_triangles_per_fiber.begin(), adjacent_triangles_per_fiber.end());

        // Convertir std::set a std::vector y almacenar en los resultados finales
        intersection[fiber] = std::vector<int>(unique_intersection_per_fiber.begin(), unique_intersection_per_fiber.end());
        adjacent_triangles[fiber] = std::vector<int>(unique_adjacent_triangles_per_fiber.begin(), unique_adjacent_triangles_per_fiber.end());
    }
    return {intersection, adjacent_triangles};
}

// Definir un punto 3D
struct Point {
    double x, y, z;

    Point operator-(const Point& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Point operator+(const Point& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Point operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
};

// Calcular la norma de un vector
double norm(const Point& p) {
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

// Normalizar un vector
Point normalize(const Point& p) {
    double n = norm(p);
    return {p.x / n, p.y / n, p.z / n};
}

// Calcular la proyección de una fibra
py::array_t<float> proyection_fiber(py::array_t<float> fiber, float tolerance = 5.0, float vox_resolution = 0.7) {
    auto f = fiber.unchecked<2>();
    int n_points = f.shape(0);

    float tolerance_vox = tolerance / vox_resolution;

    // Calcular distancia máxima entre puntos consecutivos
    float max_distance = 0.0;
    for (size_t i = 1; i < n_points; ++i) {
        Point p1 = {f(i - 1, 0), f(i - 1, 1), f(i - 1, 2)};
        Point p2 = {f(i, 0), f(i, 1), f(i, 2)};
        float d = norm(p2 - p1);
        max_distance = std::max(max_distance, d);
    }

    int points_to_add = ceil(tolerance_vox / max_distance);
    py::array_t<float> result({n_points + 2 * points_to_add, 3});
    auto r = result.mutable_unchecked<2>();

    // Proyección al inicio
    Point origin = {f(0, 0), f(0, 1), f(0, 2)};
    Point direction = normalize({f(0, 0) - f(1, 0), f(0, 1) - f(1, 1), f(0, 2) - f(1, 2)});

    for (int i = 0; i < points_to_add; ++i) {
        Point p = origin + direction * ((i + 1) * tolerance_vox / points_to_add);
        r(points_to_add - (i + 1), 0) = p.x;
        r(points_to_add - (i + 1), 1) = p.y;
        r(points_to_add - (i + 1), 2) = p.z;
    }

    // Copiar la fibra original
    for (size_t i = 0; i < n_points; ++i) {
        r(points_to_add + i, 0) = f(i, 0);
        r(points_to_add + i, 1) = f(i, 1);
        r(points_to_add + i, 2) = f(i, 2);
    }

    // Proyección al final
    origin = {f(n_points - 1, 0), f(n_points - 1, 1), f(n_points - 1, 2)};
    direction = normalize({f(n_points - 1, 0) - f(n_points - 2, 0), f(n_points - 1, 1) - f(n_points - 2, 1), f(n_points - 1, 2) - f(n_points - 2, 2)});

    for (int i = 0; i < points_to_add; ++i) {
        Point p = origin + direction * ((i + 1) * tolerance_vox / points_to_add);
        r(n_points + points_to_add + i, 0) = p.x;
        r(n_points + points_to_add + i, 1) = p.y;
        r(n_points + points_to_add + i, 2) = p.z;
    }

    return result;
}

PYBIND11_MODULE(octree_module, m) {
    py::class_<Triangle>(m, "Triangle")
        .def(py::init<py::array_t<float>, py::array_t<int>, int>())
        .def_property_readonly("vertices", &Triangle::get_vertices)
        .def_property_readonly("indices", &Triangle::get_indices)
        .def_property_readonly("normal", &Triangle::get_normal)
        .def_property_readonly("area", &Triangle::get_area)
        .def_readonly("idx", &Triangle::idx);

    py::class_<OctreeNode>(m, "Octree")
        .def(py::init<py::array_t<float>, int, int, int, float>(),
            py::arg("bounds"),
            py::arg("depth") = 0,
            py::arg("min_depth") = 5,
            py::arg("max_depth") = 7,
            py::arg("epsilon") = 0.01f)
        .def("insert_scene", &OctreeNode::insert_scene,
            py::arg("scene_vertices"),
            py::arg("scene_indices"))
        .def("subdivide", &OctreeNode::subdivide)
        .def_readonly("bounds", &OctreeNode::bounds)
        .def_readonly("triangles", &OctreeNode::triangles)
        .def_readonly("children", &OctreeNode::children)
        .def_readonly("depth", &OctreeNode::depth)
        .def_readonly("max_depth", &OctreeNode::max_depth)
        .def("calculate_normal_variance", &OctreeNode::calculate_normal_variance)
        .def("recursive_subdivision", &OctreeNode::recursive_subdivision)
        .def("recursive_subdivision_vertex", &OctreeNode::recursive_subdivision_vertex)
        .def("redistribute_triangles_vertex", &OctreeNode::redistribute_triangles_vertex)
        .def("redistribute_triangles", &OctreeNode::redistribute_triangles)
        .def("find_leaf_containing_point", &OctreeNode::find_leaf_containing_point,
            py::arg("point"))
        .def("get_adjacent_triangles", &OctreeNode::get_adjacent_triangles,
            py::arg("point"))
        .def("get_leaf_nodes", &OctreeNode::get_leaf_nodes)
        .def("fill_lookup_table", &OctreeNode::fill_lookup_table)
        .def("calculate_xyz_range", &OctreeNode::calculate_xyz_range)
        .def("interpolate_fiber", &OctreeNode::interpolate_fiber,
            py::arg("fiber"));
    m.def("is_triangle_in_bounds", &is_triangle_in_bounds, "Check if triangle is inside bounds");
    m.def("separating_axis_theorem", &separating_axis_theorem, "Separating Axis Theorem");
    m.def("moller_trumbore", &moller_trumbore, "Moller-Trumbore algorithm",
        py::arg("ray_origin"),
        py::arg("ray_direction"),
        py::arg("triangle"));
    m.def("ray_triangle_intersection", &ray_triangle_intersection, "Ray-Triangle intersection",
        py::arg("origins"),
        py::arg("directions"),
        py::arg("vertices"),
        py::arg("faces"));
    m.def("intersection_octree", &intersection_octree, "Query intersection using Octree",
        py::arg("octree"),
        py::arg("origins"),
        py::arg("directions"),
        py::arg("vertices"),
        py::arg("faces"));
    m.def("calculate_k_points", &calculate_k_points, "Calculate k points between two points",
        py::arg("p1"),
        py::arg("p2"),
        py::arg("w") = 3,
        py::arg("delta") = 1);
    m.def("generate_points_interpolated", &generate_points_interpolated, "Generate interpolated points",
        py::arg("p1"),
        py::arg("p2"),
        py::arg("k"));
    m.def("max_k_in_fiber_proyection", &max_k_in_fiber_proyection, "Calculate max k in fiber proyection",
        py::arg("fiber_projection"));
    m.def("compute_directions_fiber", &compute_directions_fiber, "Compute directions for fiber",
        py::arg("fiber"));
    m.def ("find_intersections", &find_intersections, "Find intersections",
        py::arg("octree"),
        py::arg("centered_bundle"),
        py::arg("centered_vertices"),
        py::arg("polygons_lh"));
    m.def("proyection_fiber", &proyection_fiber, "Proyection of a fiber",
        py::arg("fiber"),
        py::arg("tolerance") = 5.0,
        py::arg("vox_resolution") = 0.7);
}
