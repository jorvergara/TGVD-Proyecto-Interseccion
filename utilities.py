import octree_module
import os
import pickle
import numpy as np
import bundleTools as BT
import bundleTools3 as BT3
import time
import intersection_utils as iu

def load_data():
    """
    Carga los datos de las fibras y el mesh.
    """
    mesh_lh_path= 'meshes/lh.obj'
    bundles_path = 'tract/3Msift_t_MNI_21p.bundles'
    bundle = np.array(BT.read_bundle(bundles_path))
    vertex_lh, polygons_lh = BT.read_mesh_obj(mesh_lh_path)
    print("Number of fibers in the bundle:", len(bundle))
    print("Number of triangles in the mesh:", len(polygons_lh))
    return bundle, vertex_lh, polygons_lh

def build_octree(bounds, vertex_lh, polygons_lh):
    """
    Construye un octree con SAT a partir de los límites y el mesh.
    """
    octree = octree_module.Octree(bounds)
    octree.insert_scene(vertex_lh, polygons_lh)
    octree.recursive_subdivision()
    return octree

def build_octree_vertex(bounds, vertex_lh, polygons_lh):
    """
    Construye un octree con vértices a partir de los límites y el mesh.
    """
    octree_vertex = octree_module.Octree(bounds)
    octree_vertex.insert_scene(vertex_lh, polygons_lh)
    octree_vertex.recursive_subdivision_vertex()
    return octree_vertex

def calculate_intersections_gt(N, bundle, vertex_lh, polygons_lh, gt_path = "gt"):
    from queries import query_intersection_brute
    """
    Calcula las intersecciones ground truth.
    """    
    # Cargar o calcular intersecciones
    if  os.listdir(gt_path):
        print("Existen algunos datos de intersección.")
        selected_fibers_gt = np.load('gt/selected_fibers_gt.npy')
        with open("gt/intersection_gt.pkl", "rb") as f:
            intersection_gt = pickle.load(f)
        
    
    # Calcular nuevos datos de GT
    num_fib = len(selected_fibers_gt)
    if num_fib != N:
        print("Calculando GT...")
        selected_fibers_gt, intersection_gt = query_intersection_brute(N, bundle, vertex_lh, polygons_lh, seed=42)
        os.makedirs('gt', exist_ok=True)
        np.save('gt/selected_fibers_gt.npy', selected_fibers_gt)
        with open("gt/intersection_gt.pkl", "wb") as f:
            pickle.dump(intersection_gt, f)

    return selected_fibers_gt, intersection_gt

def create_multiple_test_bundles(N, bundle):

    def create_test_bundle(N, bundle):
        """
        Crea un subconjunto de fibras de tamaño N y lo guarda en un archivo .bundles.
        """
        # Seleccionar índices aleatorios
        random_indices = np.random.choice(len(bundle), N, replace=False)
        test_bundle = bundle[random_indices]

        # Guardar el test bundle como archivo .bundles
        output_file = f"test/test_{N}/3Msift_t_MNI_21p_bundle_{N}.bundles"
        os.makedirs(f"test/test_{N}", exist_ok=True)
        BT3.write_bundle(output_file, test_bundle)

        print(f"Guardado: {output_file}")
    
    # Crear subconjuntos de fibras
    for n in N:
        create_test_bundle(n, bundle)

# Función para medir tiempo del algoritmo de Felipe
def test_felipe(N, meshes_path):
    times_felipe = []
    for i in N:
        bundles_path = f'test/test_{i}/'
        results_path = f'results/test_{i}/'
        start_time = time.time()
        iu.intersection(meshes_path, bundles_path, results_path)
        time_felipe = time.time() - start_time
        times_felipe.append(time_felipe)
        print(f"Tiempo de ejecución Felipe para {i} fibras: {time_felipe:.4f}s")
    return times_felipe

# Función para medir tiempo del algoritmo con Octree
def test_octree(N, bounds, vertex_lh, polygons_lh, octree):
    times_octree = []
    for i in N:
        bundles_test_path = f'test/test_{i}/3Msift_t_MNI_21p_bundle_{i}.bundles'
        bundle_test = np.array(BT.read_bundle(bundles_test_path))
        centered_bundle_test = bundle_test - bounds[:3]
        centered_vertices_test = vertex_lh - bounds[:3]

        start = time.time()
        insersection_octree, _ = octree_module.find_intersections(octree, centered_bundle_test, centered_vertices_test, polygons_lh)
        time_octree = time.time() - start
        times_octree.append(time_octree)

        # Guardar resultados
        subfolder = os.path.join('results', f"test_octree{i}")
        os.makedirs(subfolder, exist_ok=True)
        output_file = os.path.join(subfolder, f"intersection_octree_{i}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(insersection_octree, f)

        print(f"Tiempo de ejecución Octree para {i} fibras: {time_octree:.4f}s")
    return times_octree

def obtener_indices_por_clase(InTri, FnTri, fib_index, fiber_class):
    
    """ En README de intersección de Felipe (intersection/README.md) para cada intersección
    se tiene una clase, donde:
    clase 1 -> L-L
    clase 2 -> L-H
    clase 3 -> H-L
    clase 4 -> H-H
    por lo que para comparar cada algoritmo dado que nosotros solo lo calculamos para el 
    hemisferio izquierdo, se tiene que hacer lo siguiente, de:

    clase 1 -> InTri FnTri (triángulos que intersectan al inicio y al final de la fibra)
    clase 2 -> InTri (triángulos que intersectan al inicio de la fibra)
    clase 3 -> FnTri (triángulos que intersectan al final de la fibra) """

    triangulos = []
    fibras = []
    for i in range(len(fiber_class)):
        if fiber_class[i] == 1:
            triangulos.append(InTri[i])
            fibras.append(fib_index[i])
            triangulos.append(FnTri[i])
            fibras.append(fib_index[i])
        elif fiber_class[i] == 2:
            triangulos.append(InTri[i])
            fibras.append(fib_index[i])
        elif fiber_class[i] == 3:
            triangulos.append(FnTri[i])
            fibras.append(fib_index[i])

    return np.array(triangulos), np.array(fibras)

