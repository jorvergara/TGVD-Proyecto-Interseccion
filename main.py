#%%
import sys
sys.path.append('build')
import timeit
import numpy as np
import matplotlib.pyplot as plt
import intersection_utils as iu
import octree_module

import subprocess as sp
import numpy as np
import bundleTools as BT
import bundleTools3 as BT3
import visualizationTools as vt
from tqdm import tqdm
import pickle
import time
import os
from fury import window, actor

mesh_lh_path= 'meshes/lh.obj'
bundles_path = 'tract/3Msift_t_MNI_21p.bundles'
bundle = np.array(BT.read_bundle(bundles_path))
vertex_lh, polygons_lh = BT.read_mesh_obj(mesh_lh_path)
print("Number of fibers in the bundle:", len(bundle))
print("Number of triangles in the mesh:", len(polygons_lh))

#%%
bounds = np.concatenate([np.min(vertex_lh, axis=0), np.max(vertex_lh, axis=0)]).astype(np.float32)
time_start = time.time()
octree = octree_module.Octree(bounds)
octree.insert_scene(vertex_lh, polygons_lh)
octree.recursive_subdivision()
construction_time = time.time() - time_start
print(f"Octree construction time: {construction_time:.4f}s")

#%%
time_start = time.time()
octree_vertex = octree_module.Octree(bounds)
octree_vertex.insert_scene(vertex_lh, polygons_lh)
octree_vertex.recursive_subdivision_vertex()
construction_time_vertex = time.time() - time_start
print(f"Octree_vertex construction time: {construction_time_vertex:.4f}s")

#%% Plotear el tiempo de construcción
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.bar(["Octree SAT", "Octree Vertex"], [construction_time, construction_time_vertex])
plt.ylabel("Tiempo de Construcción (segundos)") 
plt.title("Comparación de Tiempos de Construcción")
plt.grid(True)
plt.savefig("figures/octree_construction_time.png", dpi=300)
plt.show()
#%%
from queries import query_intersection_brute
N = 1000
selected_fibers_gt, intersection_gt = query_intersection_brute(N, bundle, vertex_lh, polygons_lh, seed=42)
#%% Save selected fibers and intersection
os.makedirs('gt', exist_ok=True)
np.save('gt/selected_fibers_gt.npy', selected_fibers_gt)
with open("gt/intersection_gt.pkl", "wb") as f:
    pickle.dump(intersection_gt, f)

#%% Load selected fibers and intersection
selected_fibers_gt = np.load('gt/selected_fibers_gt.npy')
with open("gt/intersection_gt.pkl", "rb") as f:
    intersection_gt = pickle.load(f)
#%%
#%%
centered_bundle = bundle - bounds[:3]
centered_vertices = vertex_lh - bounds[:3]

start = time.time()
insersection_vertex, adjacents_vertex = octree_module.find_intersections(octree_vertex, centered_bundle[selected_fibers_gt], centered_vertices, polygons_lh)
print(time.time() - start)
#%%
start = time.time()
insersection, adjacents = octree_module.find_intersections(octree, centered_bundle[selected_fibers_gt], centered_vertices, polygons_lh)
print(time.time() - start)

#%%
def gen_pares_intersecciones(lista_intersecciones):
    return {
        (fibra_idx, tri_idx) 
        for fibra_idx, lista_triangulos in enumerate(lista_intersecciones) 
        for tri_idx in lista_triangulos
    }
pares_intersecciones_gt = gen_pares_intersecciones(intersection_gt)
pares_intersecciones_vertex = gen_pares_intersecciones(insersection_vertex)
pares_intersecciones = gen_pares_intersecciones(insersection)

print(f"Intersecciones con octree_SAT: {len(pares_intersecciones)}")
print(f"Intersecciones con octree_vertex: {len(pares_intersecciones_vertex)}")
print(f"Intersecciones ground truth: {len(pares_intersecciones_gt)}")

#%%
plt.figure(figsize=(10, 10))

#%%
def pair_recall(set_a, set_b):
    if not set_b:
        return 0
    return len(set_a & set_b) / len(set_b)

def pair_discrepancy(set_a, set_b):
    return len(set_a - set_b) + len(set_b - set_a)

recall_intersecciones = pair_recall(pares_intersecciones, pares_intersecciones_gt)
recall_intersecciones_vertex = pair_recall(pares_intersecciones_vertex, pares_intersecciones_gt)
discrepancy_intersecciones = pair_discrepancy(pares_intersecciones, pares_intersecciones_gt)
discrepancy_intersecciones_vertex = pair_discrepancy(pares_intersecciones_vertex, pares_intersecciones_gt)

print(f"Recall intersecciones con octree_SAT: {recall_intersecciones:.4f}")
print(f"Recall intersecciones con octree_vertex: {recall_intersecciones_vertex:.4f}")
print(f"Discrepancia intersecciones con octree_SAT: {discrepancy_intersecciones}")
print(f"Discrepancia intersecciones con octree_vertex: {discrepancy_intersecciones_vertex}")

#%% Test en distintos tamaños de N
N = [10,50,100,200,500,1000,5000,10000,50000,100000,500000,1000000]
output_dir = "test"
#%%
# Generar y guardar subconjuntos
for i in N:
    # Seleccionar índices aleatorios
    random_indices = np.random.choice(len(bundle), i, replace=False)
    test_bundle = bundle[random_indices]
    
    # Crear una subcarpeta con el nombre correspondiente
    subfolder = os.path.join(output_dir, f"test_{i}")
    os.makedirs(subfolder, exist_ok=True)
    
    # Guardar el test bundle como archivo .bundles
    output_file = os.path.join(subfolder, f"3Msift_t_MNI_21p_bundle_{i}.bundles")
    BT3.write_bundle(output_file, test_bundle)
    
    print(f"Guardado: {output_file}")
#%% Intersección Felipe
times_felipe = []
meshes_path= 'meshes/'
for i in N: 
    bundles_path = f'test/test_{i}/'  
    results_path = f'results/test_{i}/'
    star_time = time.time()
    iu.intersection(meshes_path, bundles_path, results_path)
    time_felipe = time.time() - star_time
    times_felipe.append(time_felipe)
    print(f"Tiempo de ejecución Felipe: {time_felipe:.4f}s")

#%% 
times_octree = []

for i in N:
    bundles_test_path = f'test/test_{i}/3Msift_t_MNI_21p_bundle_{i}.bundles'
    bundle_test = np.array(BT.read_bundle(bundles_test_path))
    centered_bundle_test = bundle_test - bounds[:3]
    centered_vertices_test = vertex_lh - bounds[:3]

    start = time.time()
    insersection_octree, adjacents_octree = octree_module.find_intersections(octree, centered_bundle_test, centered_vertices_test, polygons_lh)
    time_octree = time.time() - start
    times_octree.append(time_octree)

    # Crear una subcarpeta con el nombre correspondiente
    subfolder = os.path.join('results', f"test_octree{i}")
    os.makedirs(subfolder, exist_ok=True)

    # Guardar intersecciones como pkl
    output_file = os.path.join(subfolder, f"intersection_octree_{i}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(insersection_octree, f)

    print(f"Tiempo de ejecución Octree: {time_octree:.4f}s")
#%%
plt.figure(figsize=(10, 10))
plt.plot(N, times_felipe, label="Felipe")
plt.plot(N, times_octree, label="Octree")
plt.xlabel("Número de Fibras")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Comparación de Tiempos de Ejecución")
plt.legend()
plt.grid(True)
plt.savefig("figures/times_comparison.png", dpi=300)
plt.show()
#%%
# clase 1 -> InTri FnTri (L-L)
# clase 2 -> InTri (L-H)
# clase 3 -> FnTri (H-L)

def obtener_indices_por_clase(InTri, FnTri, fib_index, fiber_class):
    # Devuelve los resultados según indicados en README (carpeta intersection) de implementación de Felipe
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

def inclusion_metric(set_a, set_b):
    """
    Calcula la métrica de inclusión de set_a en set_b
    set_a: Conjunto de referencia (esperado)
    set_b: Conjunto evaluado (predicho)
    
    Retorna:
    - 1 si set_a está completamente contenido en set_b
    - Valor entre 0 y 1 según el grado de inclusión
    """
    if not set_a:
        return 1  # Si A está vacío, consideramos que está contenido
    return len(set_a & set_b) / len(set_a)

def discrepancy_metric(set_a, set_b):
    """
    Calcula la métrica de discrepancia entre set_a y set_b
    set_a: Conjunto de referencia (esperado)
    set_b: Conjunto evaluado (predicho)
    
    Retorna:
    - 0 si los conjuntos son idénticos
    - Valor mayor a 0 si los conjuntos difieren
    """
    return len(set_a) - len(set_a & set_b) 
#%% Cargar datos de intersección Felipe
inclusion_metrics = []
discrepancy_metrics = []
intersections_felipe = []
intersection_octree = []

for i in N:
    results_path = f'results/test_{i}/'
    intersection_file = results_path + f'3Msift_t_MNI_21p_bundle_{i}.intersectiondata'
    InTri, FnTri, InPoints, FnPoints, fib_index, fiber_class = iu.read_intersection(intersection_file)

    # Llamar a la función
    triangulos_felipe, fibras_felipe = obtener_indices_por_clase(InTri, FnTri, fib_index, fiber_class)
    triangulos_felipe = triangulos_felipe.tolist()
    fibras_felipe = fibras_felipe.tolist()
    intersection_pares_felipe = {(fibra, triangulo) for fibra, triangulo in zip(fibras_felipe, triangulos_felipe)}
    # Cargar datos de intersección Octree
    result_path_octree = f'results/test_octree{i}/'
    intersection_file_octree = result_path_octree + f'intersection_octree_{i}.pkl'
    with open(intersection_file_octree, "rb") as f:
        insersection_octree = pickle.load(f)
    intersection_pares_octree = gen_pares_intersecciones(insersection_octree)

    inclusion_metric_felipe = inclusion_metric(intersection_pares_felipe, intersection_pares_octree)
    discrepancy_metric_felipe = discrepancy_metric(intersection_pares_felipe, intersection_pares_octree)

    inclusion_metrics.append(inclusion_metric_felipe)
    discrepancy_metrics.append(discrepancy_metric_felipe)
    intersections_felipe.append(len(intersection_pares_felipe))
    intersection_octree.append(len(intersection_pares_octree))
#%% Comparación de detección de intersecciones entre Felipe y Octree
plt.figure(figsize=(10, 10))
plt.plot(N, intersections_felipe, label="Total de intersecciones Felipe")
plt.plot(N, intersection_octree, label="Total de intersecciones Octree")
plt.xlabel("Número de Fibras")
plt.ylabel("Número de Intersecciones")
plt.title("Comparación de Intersecciones")
plt.legend()
plt.grid(True)
plt.savefig("figures/intersections_comparison.png", dpi=300)
plt.show()

#%% Comparación con dibras de felipe que no estan en el octree (fp)
plt.figure(figsize=(10, 10))
plt.plot(N, intersections_felipe, label="Total de intersecciones Felipe")
plt.plot(N, discrepancy_metrics, label="Discrepancia")
plt.xlabel("Número de Fibras")
plt.ylabel("Número de Intersecciones")
plt.title("Comparación de Intersecciones")
plt.legend()
plt.grid(True)
plt.savefig("figures/discrepancia.png", dpi=300)
plt.show()
#%%
plt.figure(figsize=(10, 10))
plt.plot(N, inclusion_metrics, label="Inclusión")
plt.xlabel("Número de Fibras")
plt.ylabel("Inclusión")
plt.title("Métrica de Inclusión")
plt.legend()
plt.grid(True)
plt.savefig("figures/inclusion.png", dpi=300)

#%%
# Intersecciones del algoritmo de felipe que no están en octree (podrían ser falsos positivos)
no_f = intersection_pares_felipe - intersection_pares_octree
fibras_no_f = np.array([f for f, t in no_f])
triangles_no_f = np.array([t for f, t in no_f])

# Intersecciones que están en ambos algoritmos (podrían ser tp)
intersec = intersection_pares_felipe & intersection_pares_octree
fibras_intersec = np.array([f for f, t in intersec])
triangles_intersec = np.array([t for f, t in intersec])

# Intersecciones del algoritmo de octree (todas las intersecciones que detecta el octree)
fibras_octree = np.array([f for f, t in intersection_pares_octree])
triangles_octree = np.array([t for f, t in intersection_pares_octree])
#%%
_min = 120
_max = 140
bool_fibers = (_min < fibras_octree) & (fibras_octree < _max)
scene = window.Scene()
surface_actor = actor.surface(centered_vertices_test, polygons_lh)
# surface_actor.GetProperty().SetOpacity(0.2)
scene.add(surface_actor)

bundle_actor = actor.line(centered_bundle_test[fibras_octree[bool_fibers]], linewidth=1)
scene.add(bundle_actor)

adjacent_actor = actor.surface(centered_vertices_test, polygons_lh[triangles_octree[bool_fibers]],
                                colors=np.array(len(vertex_lh)*[[1, 0, 0]]))
scene.add(adjacent_actor)

window.show(scene)
#%%
_min = 20
_max = 50
bool_fibers = (_min < fibras_intersec) & (fibras_intersec < _max)
scene = window.Scene()
surface_actor = actor.surface(centered_vertices_test, polygons_lh)
surface_actor.GetProperty().SetOpacity(0.2)
# scene.add(surface_actor)

bundle_actor = actor.line(centered_bundle_test[fibras_octree[bool_fibers]], linewidth=1)
scene.add(bundle_actor)

adjacent_actor = actor.surface(centered_vertices_test, polygons_lh[triangles_octree[bool_fibers]],
                                colors=np.array(len(vertex_lh)*[[1, 0, 0]]))
scene.add(adjacent_actor)

window.show(scene)
#%%
_min = 100
_max = 120
bool_fibers = (_min < fibras_no_f) & (fibras_no_f < _max)
scene = window.Scene()
surface_actor = actor.surface(centered_vertices_test, polygons_lh)
surface_actor.GetProperty().SetOpacity(0.6)
# scene.add(surface_actor)

bundle_actor = actor.line(centered_bundle_test[fibras_no_f[bool_fibers]], linewidth=1)
scene.add(bundle_actor)

adjacent_actor = actor.surface(centered_vertices_test, polygons_lh[triangles_no_f[bool_fibers]],
                                colors=np.array(len(vertex_lh)*[[1, 0, 0]]))
scene.add(adjacent_actor)

window.show(scene)

