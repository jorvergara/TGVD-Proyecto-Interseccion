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
octree.insert_scene(vertex_lh, polygons_lh[:1000])
octree.recursive_subdivision()
#%%
# from queries import query_intersection_brute, query_intersection_octree

# # Valores de N para probar
# N_values = [10, 50, 100, 200, 500]

# # Inicializar resultados
# brute_times = []
# octree_times = []

# # Simulación con los datos
# for N in N_values:
#     print(f"Procesando N={N}...")

#     # Medir tiempo para query_intersection_brute
#     start_time = timeit.default_timer()
#     selected_fibers, intersection = query_intersection_brute(N, bundle, vertex_lh, polygons_lh, seed=42)
#     brute_time = timeit.default_timer() - start_time
#     brute_times.append(brute_time)

#     # Medir tiempo para query_intersection_octree
#     start_time = timeit.default_timer()
#     intersection_octree = query_intersection_octree(octree, N, bundle, vertex_lh, polygons_lh, seed=42)
#     octree_time = timeit.default_timer() - start_time
#     octree_times.append(octree_time)

#     print(f"Brute time: {brute_time:.4f}s, Octree time: {octree_time:.4f}s")
# #%% Graficar resultados
# import matplotlib.pyplot as plt 
# plt.figure(figsize=(10, 6))
# plt.plot(N_values, brute_times, label="Brute Force", marker="o")
# plt.plot(N_values, octree_times, label="Octree", marker="s")
# plt.xlabel("N (Número de Fibras)")
# plt.ylabel("Tiempo de Ejecución (segundos)")
# plt.title("Comparación de Tiempos de Ejecución")
# plt.legend()
# plt.grid(True)
# plt.show()

# #%%
# # Valores de N para probar
# N_values_log = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]

# # Inicializar resultados
# octree_times_log = []

# # Medir tiempos para Octree
# for N in N_values_log:
#     print(f"Procesando N={N} para Octree...")

#     # Medir tiempo para query_intersection_octree
#     start_time = timeit.default_timer()
#     intersection_octree = query_intersection_octree(octree, N, bundle, vertex_lh, polygons_lh, seed=42)
#     octree_time_log = timeit.default_timer() - start_time
#     octree_times_log.append(octree_time_log)

#     print(f"Octree time: {octree_time_log:.4f}s")

# #%%
# plt.figure(figsize=(10, 6))
# plt.plot(N_values_log, octree_times_log, label="Octree", marker="s")
# plt.xlabel("N (Número de Fibras)")
# plt.ylabel("Tiempo de Ejecución (segundos)")
# plt.title("Tiempos de Ejecución Octree")
# plt.legend()
# plt.grid(True)
# plt.show()

# #%%
# selected_fibers = intersection_octree[0]
# intersection = sum(intersection_octree[1],[])
# #%%

# from fury import window, actor
# scene = window.Scene()
# surface_actor = actor.surface(vertex_lh, polygons_lh)
# surface_actor.GetProperty().SetOpacity(0.5)
# scene.add(surface_actor)

# # bundle_actor = actor.line(selected_fibers, linewidth=2)
# # scene.add(bundle_actor)

# # point_actor = actor.point([point], colors=[(1, 0, 0)], point_radius=1)
# # scene.add(point_actor)

# adjacen_triangles_actor = actor.surface(vertex_lh, polygons_lh[intersection], colors=np.array(len(vertex_lh)*[[1, 0, 0]]))
# scene.add(adjacen_triangles_actor)
# window.show(scene)
# # # %%

# # %%
