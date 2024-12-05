#%%
import sys
sys.path.append('build')
import timeit
import numpy as np
import matplotlib.pyplot as plt

import octree_module

import subprocess as sp
import numpy as np
import bundleTools as BT
import bundleTools3 as BT3
import visualizationTools as vt
from tqdm import tqdm

mesh_lh_path= 'meshes/lh.obj'
bundles_path = 'tract/3Msift_t_MNI_21p.bundles'
bundle = np.array(BT.read_bundle(bundles_path))
vertex_lh, polygons_lh = BT.read_mesh_obj(mesh_lh_path)
print("Number of fibers in the bundle:", len(bundle))
print("Number of triangles in the mesh:", len(polygons_lh))

tract_path = 'fasciculos/reoriented/lh_RMF-RMF_1.bundles'
tract_bundle = BT.read_bundle(tract_path)
print('Number of points in the mesh:', len(tract_bundle))

#%%
bounds = np.concatenate([np.min(vertex_lh, axis=0), np.max(vertex_lh, axis=0)]).astype(np.float32)
octree = octree_module.Octree(bounds)
octree.insert_scene(vertex_lh, polygons_lh)
octree.recursive_subdivision()

#%%
def get_leaf_nodes(octree):
    """
    Devuelve una lista de todos los nodos hoja en el Octree.

    Args:
        octree (OctreeNode): Nodo raíz del Octree.

    Returns:
        list: Lista de nodos hoja.
    """
    if not octree.children:
        return [octree]  # Si no tiene hijos, es un nodo hoja

    # Recorrer recursivamente los hijos y acumular los nodos hoja
    leaf_nodes = []
    for child in octree.children:
        leaf_nodes.extend(get_leaf_nodes(child))
    return leaf_nodes

leaf_nodes = get_leaf_nodes(octree)

print(f"Total de nodos hoja: {len(leaf_nodes)}")
#%%
bounds = np.concatenate([np.min(vertex_lh, axis=0), np.max(vertex_lh, axis=0)]).astype(np.float32)

table_look_up = np.zeros((2**7, 2**7, 2**7), dtype=np.int32)
x_range = (bounds[3] - bounds[0])/2**7
y_range = (bounds[4] - bounds[1])/2**7
z_range = (bounds[5] - bounds[2])/2**7
from tqdm import tqdm
# for idx in tqdm(range(len(leaf_nodes))):
for idx in tqdm(range(len(leaf_nodes))):  
    leaf_node_i = leaf_nodes[idx]
    centered_bounds = leaf_node_i.bounds.copy()
    centered_bounds[0] -= bounds[0]
    centered_bounds[1] -= bounds[1]
    centered_bounds[2] -= bounds[2]
    centered_bounds[3] -= bounds[0]
    centered_bounds[4] -= bounds[1]
    centered_bounds[5] -= bounds[2]

    x_i = [np.round((centered_bounds[0])/x_range + i).astype(np.int32) for i in range(2**(7-leaf_node_i.depth))]
    y_i = [np.round((centered_bounds[1])/y_range + i).astype(np.int32) for i in range(2**(7-leaf_node_i.depth))]
    z_i = [np.round((centered_bounds[2])/z_range + i).astype(np.int32) for i in range(2**(7-leaf_node_i.depth))]

    for x in x_i:
        for y in y_i:
            for z in z_i:
                table_look_up[x, y, z] = idx
#%%
from queries import compute_directions
centered_bundle = bundle - bounds[:3]
centered_origin, centered_direction = compute_directions(centered_bundle[:1000])
centered_vertices = vertex_lh - bounds[:3]
#%%

dx = [-1,0,1]
adjacent_nodes = [[] for _ in range(len(leaf_nodes))]
xyz_range = np.array([x_range, y_range, z_range])
intersection = [[] for _ in range(len(centered_bundle))]
adjacent_triangles = [[] for _ in range(len(centered_bundle))]

# for fiber in tqdm(range(centered_origin.shape[0])[:2]):

fiber = 12
intersenction_per_point = []
adjacent_triangles_per_fiber = []
for point in range(centered_origin.shape[1]):
    centered_point = centered_origin[fiber, point]
    
    p_xi, p_yi, p_zi = np.floor((centered_point)/xyz_range).astype(np.int32)
   
    if p_xi < 0 or p_yi < 0 or p_zi < 0 or p_xi >= 2**7 or p_yi >= 2**7 or p_zi >= 2**7:
        continue
    
    if not adjacent_nodes[table_look_up[p_xi, p_yi, p_zi]]:
        adjacent_idx = [[p_xi + i, p_yi + j, p_zi + k]
                        for i in dx for j in dx for k in dx
                        if 0 <= p_xi + i < 2**7 and 0 <= p_yi + j < 2**7 and 0 <= p_zi + k < 2**7]
        
        adjacent_idx = np.array(adjacent_idx)
        adjacent_nodes_i = list(set(table_look_up[adjacent_idx[:,0], adjacent_idx[:,1], adjacent_idx[:,2]]))
        
        
        if adjacent_nodes_i:
            adjacent_nodes[table_look_up[p_xi, p_yi, p_zi]] = adjacent_nodes_i
        else:
            adjacent_nodes[table_look_up[p_xi, p_yi, p_zi]] = ["X"]

    # adjacent_triangles_i = sum([leaf_nodes[idx].triangles for idx in adjacent_nodes_i], [])
    # adjacent_triangles_i = list(set([i.idx for i in adjacent_triangles_i]))

        # if adjacent_triangles_i:
            # intersenction_per_point = [idx for idx in adjacent_triangles_i 
            #                            if octree_module.moller_trumbore(centered_point, centered_direction[fiber, point], centered_vertices[polygons_lh[idx]])]
            # print(intersenction_per_point)
    # adjacent_triangles_per_fiber = list(set(adjacent_triangles_per_fiber))
        # adjacent_triangles_per_fiber = set(adjacent_triangles_per_fiber)
        #     # if adjacent_triangles_i:
        #     #     for idx in adjacent_triangles_i:
        #     #         centered_point_direction = centered_direction[fiber, point]

        #     #         # if octree_module.moller_trumbore(centered_point, centered_point_direction, centered_vertices[polygons_lh[idx]]):
        #     #         #     intersenction_per_point.append(idx)
        #     #         #     print("interseccion")
#%%
print(adjacent_nodes[table_look_up[p_xi, p_yi, p_zi]])   
# print(adjacent_triangles_per_fiber)
#%%
print(adjacent_triangles_per_fiber)


# adjacent_nodes = [[] for _ in range(len(leaf_nodes))]

# for idx in tqdm(range(len(leaf_nodes))):
#%%
print(adjacent_nodes[1000:2000])
#%%
A = ["X"]
if not A:
    print("hola")
#%%
print(centered_bundle.shape)
#%%
print(table_look_up[7,49,83])

#%%
a = np.array([10,20,30,40,50,60]) - np.array([10,20,30]) 
#%%
print(list(range(1)))
#%%
leaf_nodes[200].depth
#%%
from queries import query_intersection_brute, query_intersection_octree
#%%
# Valores de N para probar
N_values = [10, 50, 100, 200, 500]

# Inicializar resultados
brute_times = []
octree_times = []

# Simulación con los datos
for N in N_values:
    print(f"Procesando N={N}...")

    # Medir tiempo para query_intersection_brute
    start_time = timeit.default_timer()
    selected_fibers, intersection = query_intersection_brute(N, bundle, vertex_lh, polygons_lh, seed=42)
    brute_time = timeit.default_timer() - start_time
    brute_times.append(brute_time)

    # Medir tiempo para query_intersection_octree
    start_time = timeit.default_timer()
    intersection_octree = query_intersection_octree(octree, N, bundle, vertex_lh, polygons_lh, seed=42)
    octree_time = timeit.default_timer() - start_time
    octree_times.append(octree_time)

    print(f"Brute time: {brute_time:.4f}s, Octree time: {octree_time:.4f}s")
#%% Graficar resultados
import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 6))
plt.plot(N_values, brute_times, label="Brute Force", marker="o")
plt.plot(N_values, octree_times, label="Octree", marker="s")
plt.xlabel("N (Número de Fibras)")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Comparación de Tiempos de Ejecución")
plt.legend()
plt.grid(True)
plt.savefig("figures/octree_vs_brute_time.png", dpi=300)
plt.show()

#%%
# Valores de N para probar
N_values_log = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

# Inicializar resultados
octree_times_log = []

# Medir tiempos para Octree
for N in N_values_log:
    print(f"Procesando N={N} para Octree...")

    # Medir tiempo para query_intersection_octree
    start_time = timeit.default_timer()
    intersection_octree = query_intersection_octree(octree, N, bundle, vertex_lh, polygons_lh, seed=42)
    octree_time_log = timeit.default_timer() - start_time
    octree_times_log.append(octree_time_log)

    print(f"Octree time: {octree_time_log:.4f}s")

#%%
plt.figure(figsize=(10, 6))
plt.plot(N_values_log, octree_times_log, label="Octree", marker="s")
plt.xlabel("N (Número de Fibras)")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Tiempos de Ejecución Octree")
plt.legend()
plt.grid(True)
plt.savefig("figures/octree_time.png", dpi=300)
plt.show()

#%%
selected_fibers = intersection_octree[0]
# intersection = sum(intersection_octree[1],[])
#%%
fiber = bundle[29]
point = fiber[8]
adjacent_triangles = octree.get_adjacent_triangles(point)
#%%
from fury import window, actor
scene = window.Scene()
surface_actor = actor.surface(centered_vertices, polygons_lh)
surface_actor.GetProperty().SetOpacity(0.2)
scene.add(surface_actor)

bundle_actor = actor.line([centered_bundle[12]], linewidth=3)
scene.add(bundle_actor)

# point_actor = actor.point([centered_point], colors=[(1, 1, 0)], point_radius=1)
# scene.add(point_actor)

adjacen_triangles_actor = actor.surface(centered_vertices, polygons_lh[adjacent_triangles_per_fiber], colors=np.array(len(vertex_lh)*[[1, 0, 0]]))
scene.add(adjacen_triangles_actor)

scene.set_camera(
    position = (2.10, 152.25, 55.63),
    focal_point = (106.56, 91.18, 87.19),
    view_up = (0.08, -0.35, -0.93)
)
window.show(scene)

# %%
print(adjacent_triangles_per_fiber)
