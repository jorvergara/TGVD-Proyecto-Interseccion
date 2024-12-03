# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:31:46 2024

@author: User
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%%
import subprocess as sp
from time import time
import numpy as np
from BundleTools import bundleTools as BT
from BundleTools import bundleTools3 as BT3
import visualizationTools as vt

### Leer
mesh_lh_path= 'meshes/lh.obj'
mesh_rh_path= 'meshes/rh.obj'
bundles_path = 'tract/3Msift_t_MNI_21p.bundles'

bundle = BT.read_bundle(bundles_path)
vertex_lh, polygons_lh = BT.read_mesh_obj(mesh_lh_path)
vertex_rh, polygons_rh = BT.read_mesh_obj(mesh_rh_path)

#%%
# class Point:
#     def __init__(self, origin, positive_direction, negative_direction):
#         self.origin = origin
#         self.positive_direction = positive_direction
#         self.negative_direction = negative_direction

#     def __repr__(self):
#         return f"Point(origin={self.origin}, positive={self.positive_direction}, negative={self.negative_direction})"

#%%
from tqdm import tqdm
from Octree import *
Triangles = []
for i in tqdm(range(len(polygons_lh))):
    Triangles.append(Triangle(vertex_lh[polygons_lh[i]], polygons_lh[i], i))

bounds = list(vertex_lh.min(axis=0).astype('float32')) + list(vertex_lh.max(axis=0).astype('float32'))
octree = OctreeNode(bounds, min_depth=5, max_depth=9)
octree.insert_scene(Triangles)
octree.recursive_subdivision()
#%%

from queries import *
import time

# Valores de N para probar
N_values = [10, 50, 100, 500, 1000]

# Inicializar resultados
brute_times = []
octree_times = []

# Simulación con los datos
for N in N_values:
    print(f"Procesando N={N}...")

    # Medir tiempo para query_intersection_brute
    start_time = time.time()
    selected_fibers, intersection = query_intersection_brute(N, bundle, vertex_lh, polygons_lh, seed=42)
    brute_time = time.time() - start_time
    brute_times.append(brute_time)

    # Medir tiempo para query_intersection_octree
    start_time = time.time()
    intersection_octree = query_intersection_octree(octree, N, bundle, vertex_lh, polygons_lh, seed=42)
    octree_time = time.time() - start_time
    octree_times.append(octree_time)

    print(f"Brute time: {brute_time:.4f}s, Octree time: {octree_time:.4f}s")
#%%
# Graficar resultados
import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 6))
plt.plot(N_values, brute_times, label="Brute Force", marker="o")
plt.plot(N_values, octree_times, label="Octree", marker="s")
plt.xlabel("N (Número de Fibras)")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Comparación de Tiempos de Ejecución")
plt.legend()
plt.grid(True)
plt.show()


#%%
# Valores de N para probar
N_values = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

# Inicializar resultados
octree_times_log = []

# Medir tiempos para Octree
for N in N_values:
    print(f"Procesando N={N} para Octree...")

    # Medir tiempo para query_intersection_octree
    start_time = time.time()
    intersection_octree = query_intersection_octree(octree, N, bundle, vertex_lh, polygons_lh, seed=42)
    octree_time_log = time.time() - start_time
    octree_times_log.append(octree_time_log)

    print(f"Octree time: {octree_time:.4f}s")
#%%
# Transformar N_values a log(N) para graficar
log_N_values = np.log(N_values)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(N_values, octree_times_log, label="Octree Time (Log Scale)", marker="o")
plt.xlabel("log(N) (Logaritmo del Número de Fibras)")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.title("Comportamiento Logarítmico del Octree")
# plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()
#%%
from fury import actor, window
n = 99
scene = window.Scene()

surface_actor = actor.surface(vertex_lh, polygons_lh)
surface_actor.GetProperty().SetOpacity(0.4)
scene.add(surface_actor)

surface_actor = actor.surface(vertex_lh, polygons_lh[intersection[n]], colors = np.array([[1,0,0]]*len(vertex_lh)))
scene.add(surface_actor)

bundle_actor = actor.line([selected_fibers[n]], linewidth=4)
scene.add(bundle_actor)

window.show(scene)

#%%
def separating_axis_theorem(triangle, bbox):
    """
    Verifica si un triángulo y una bounding box se intersectan usando SAT.
    
    Args:
        triangle (np.ndarray): Vértices del triángulo de forma (3, 3).
        bbox (np.ndarray): Bounding box definida como [[x_min, y_min, z_min, x_max, y_max, z_max]].
    
    Returns:
        bool: True si se intersectan, False si no.
    """
    # Descomponer entrada
    v0,v1,v2 = triangle
    
    bbox_min = bbox[:3]
    bbox_max = bbox[-3:]
    
    # Función auxiliar para proyecciones
    def project(points, axis):
        return np.dot(points, axis)
    
    # 1. Ejes de la bounding box
    box_axes = np.eye(3)  # [1, 0, 0], [0, 1, 0], [0, 0, 1]

    # 2. Ejes del triángulo
    edges = [v1 - v0, v2 - v1, v0 - v2]
    triangle_normal = np.cross(edges[0], edges[1])  # Normal del triángulo

    # 3. Ejes de los productos cruzados (triángulo x caja)
    cross_axes = [np.cross(edge, axis) for edge in edges for axis in box_axes]

    # Combinar todos los ejes relevantes
    axes = box_axes.tolist() + [triangle_normal] + cross_axes
    # print(triangle_normal.shape, np.shape(axes))
    # 4. Probar cada eje
    for axis in axes:
        if np.linalg.norm(axis) < 1e-6:  # Ignorar ejes degenerados
            continue
        axis = axis / np.linalg.norm(axis)  # Normalizar eje
        
        # Proyección del triángulo en el eje
        # print(np.shape(axis))
        triangle_proj = project(triangle, axis)
        # print(triangle_proj)
        triangle_min, triangle_max = triangle_proj.min(), triangle_proj.max()
        
        # Proyección de la caja en el eje
        box_corners = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]],
                                [bbox_min[0], bbox_min[1], bbox_max[2]],
                                [bbox_min[0], bbox_max[1], bbox_min[2]],
                                [bbox_min[0], bbox_max[1], bbox_max[2]],
                                [bbox_max[0], bbox_min[1], bbox_min[2]],
                                [bbox_max[0], bbox_min[1], bbox_max[2]],
                                [bbox_max[0], bbox_max[1], bbox_min[2]],
                                [bbox_max[0], bbox_max[1], bbox_max[2]]])
        box_proj = project(box_corners, axis)
        print(box_proj.min())
        box_min, box_max = box_proj.min(), box_proj.max()

        # Verificar si las proyecciones no se solapan
        if triangle_max < box_min or box_max < triangle_min:
            return False  # Separación encontrada

    return True  # No hay separación, las geometrías se intersectan

# separating_axis_theorem(vertex_lh[polygons_lh][0], bbox)
separating_axis_theorem(triangles[1], bbox)
#%%
import numpy as np

def separating_axis_theorem(triangle, bbox):
    """
    Verifica si un triángulo y una bounding box se intersectan usando SAT.
    
    Args:
        triangle (np.ndarray): Vértices del triángulo de forma (3, 3).
        bbox (np.ndarray): Bounding box definida como [[x_min, y_min, z_min, x_max, y_max, z_max]].
    
    Returns:
        bool: True si se intersectan, False si no.
    """
    # Descomponer entrada
    v0,v1,v2 = np.split(triangle, 3, axis = 1)
    
    bbox_min = bbox[:3]
    bbox_max = bbox[-3:]
    
    # Función auxiliar para proyecciones
    def project(points, axis):
        return np.dot(points, axis)
    
    # 1. Ejes de la bounding box
    box_axes = np.eye(3)  # [1, 0, 0], [0, 1, 0], [0, 0, 1]

    # 2. Ejes del triángulo
    edges = [v1 - v0, v2 - v1, v0 - v2]
    triangle_normal = [np.cross(edges[0][i], edges[1][i]) for i in range(len(triangle))]  # Normal del triángulo
    
    # 3. Ejes de los productos cruzados (triángulo x caja)
    
    cross_axes = [[np.cross(edge[i], axis) for edge in edges for axis in box_axes] for i in range(len(triangle))]
    
    axes = [box_axes.tolist() + [triangle_normal[i][0]] + cross_axes[i][0] for i in  range(len(triangle))]
    
    print(np.shape(cross_axes[0][0]))

    # print(np.shape(box_axes), triangle_normal.shape, np.shape(cross_axes))
    # Combinar todos los ejes relevantes
    # axes = np.concatenate([box_axes, triangle_normal, cross_axes], axis = 1)
    
    # box_corners = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]],
    #                         [bbox_min[0], bbox_min[1], bbox_max[2]],
    #                         [bbox_min[0], bbox_max[1], bbox_min[2]],
    #                         [bbox_min[0], bbox_max[1], bbox_max[2]],
    #                         [bbox_max[0], bbox_min[1], bbox_min[2]],
    #                         [bbox_max[0], bbox_min[1], bbox_max[2]],
    #                         [bbox_max[0], bbox_max[1], bbox_min[2]],
    #                         [bbox_max[0], bbox_max[1], bbox_max[2]]])
    
    # # 4. Probar cada eje
    # # for axis in axes:
    # #     if np.linalg.norm(axis) < 1e-6:  # Ignorar ejes degenerados
    # #         continue
    # # axes_degenerated = np.linalg.norm(axes, axis = 2, keepdims=True)<1e-6
    # norms = np.linalg.norm(axes, axis = 2, keepdims=True)
    # norms = np.where(norms == 0, 1, norms)
    # axes = axes / norms  # Normalizar eje

    # # print(triangle.shape)
    # # Proyección del triángulo en el eje

    # triangle_proj = [(triangle * axes[:,i][:,np.newaxis]).sum(axis = 1)[:, np.newaxis] for i in range(axes.shape[1])]
    
    # triangle_proj = np.concatenate(triangle_proj, axis = 1)
    
    # triangle_min, triangle_max = triangle_proj.min(axis = 2), triangle_proj.max(axis = 2)
    
    
    # # Proyección de la caja en el eje
    # # box_corners = np.array([box_corners]*len(triangle))
    # # i = 0
    # # a = 
    # # print(np.shape(a))
    # box_proj = [(box_corners * axes[:,i][:,np.newaxis]).sum(axis = 2)[:, np.newaxis] for i in range(axes.shape[1])]
    # box_proj = np.concatenate(box_proj, axis = 1)
    # box_min, box_max = box_proj.min(axis = 2), box_proj.max(axis = 2)
    
    
    # # Verificar si las proyecciones no se solapan
    # overlaps = ~((triangle_max < box_min) | (triangle_min > box_max))  # (n, num_axes)
    # print(overlaps)
    # return overlaps.all(axis=1)  # True si hay intersección en todos los ejes


    # return True  # No hay separación, las geometrías se intersectan

bbox = np.concat([vertex_lh.min(axis=0), vertex_lh.max(axis=0)])
separating_axis_theorem(vertex_lh[polygons_lh][:10], bbox)

#%%
# Triángulos (n=5, 3 vértices, 3 coordenadas)
triangles = np.array([[[0,0,0],[0,0.5,0],[0,0,0.5]],
                      [[10,10,10],[11,11,11],[12,12,12]]])

# Bounding box
bbox = np.array([0, 0, 0, 1, 1, 1]) 

# Verificar intersecciones
intersections = separating_axis_theorem(triangles, bbox)
intersections

    
#%%

#%%
i,j = 9,0
indices = np.array(ray_triangle_intersection_vector(origen[i,j], direction[i,j], vertex_lh[polygons_lh[adjacent_triangles[i][j]]])).flatten()
adjacent_triangles[i][j][indices]

#%%
a = []
a.extend(np.array([1]).tolist())
a
#%%

    #%%
intersection
#%%
n,j = 2,1
from fury import actor, window
scene = window.Scene()
surface_actor = actor.surface(vertex_lh, polygons_lh)
surface_actor.GetProperty().SetOpacity(0.4)
scene.add(surface_actor)

surface_actor = actor.surface(vertex_lh, polygons_lh[adjacent_triangles[n][j]], colors = np.array([[1,0,0]]*len(vertex_lh)))
scene.add(surface_actor)

bundle_actor = actor.line([bundle[n]], linewidth=4)
scene.add(bundle_actor)

point_actor = actor.point([bundle[n][j]], colors = np.array((1,0,0)), point_radius=1)
scene.add(point_actor)

window.show(scene)
#%%
from SCM import spherical_conformal_map
import trimesh

def vertices_centered(vertices):
    return vertices - vertices.mean(axis = 0)

vertices = vertex_lh.copy()
faces = polygons_lh.copy()

# Cargar malla
mesh = trimesh.Trimesh(vertices = vertices, faces = faces)  # Reemplaza con tu archivo
# spherical_vertices = spherical_conformal_map(vertices_centered(mesh.vertices), mesh.faces)
# np.save('spherical_vertices.npy', spherical_vertices)
#%%
spherical_vertices = np.load('spherical_vertices.npy')
#%%
# Suavizar e inflar el mallado
smoothed_vertices = smooth_vertices(
    vertices_centered(vertices),
    faces,
    iterations=10,   # Más iteraciones para mayor suavidad
    alpha=0.6              # Peso del suavizado
)

#%%
from fury import actor,window, io 
n = 2
j = -5
point = extended_fiber[j]
adjacent_triangles = octree.get_adjacent_triangles(point)
adjacent_triangles.shape
#%%
# Crear la escena
scene = window.Scene()

# Agregar las fibras (tractografía)
stream_actor = actor.line([fiber], linewidth=5)
scene.add(stream_actor)

stream_actor = actor.line([extended_fiber], colors = (1,0,0), opacity = 0.8)
scene.add(stream_actor)
# Agregar punto
point_actor = actor.point([point], colors=(0, 1, 0), point_radius=0.5)
scene.add(point_actor)
# Cargar y agregar el objeto de la superficie izquierda
surface_actor = actor.surface(vertex_lh, polygons_lh)
surface_actor.GetProperty().SetOpacity(.6)
scene.add(surface_actor)
# Agregar triangulos adjacentes
vertices = adjacent_triangles.reshape(-1, 3)  # Aplanar triángulos a una lista de vértices
faces = np.arange(len(vertices)).reshape(-1, 3)  # Cada triángulo usa tres vértices consecutivos
colors = np.array([[1.0, 0.0, 0.0]] * len(vertices))
adjacent_triangles_actor =  actor.surface(vertices, faces, colors = colors)
scene.add(adjacent_triangles_actor)
window.show(scene)
#%%
n = 2
fiber = bundle[n]

num_extension_points = 5

# Calcular el vector dirección para la extensión en el extremo inicial
direction_start = fiber[1] - fiber[0]

extended_start = [fiber[0] - direction_start * (i + 1) for i in range(num_extension_points)]
extended_start = np.array(extended_start)

# Calcular el vector dirección para la extensión en el extremo final
direction_end = fiber[-1] - fiber[-2]
extended_end = [fiber[-1] + direction_end * (i + 1) for i in range(num_extension_points)]
extended_end = np.array(extended_end)

extended_fiber = np.vstack([extended_start, fiber, extended_end])

#%%
def extend_fiber(fiber, num_extension_points=2):
    """
    Extiende una fibra añadiendo puntos en los extremos.
    :param fiber: Array de forma (21, 3) que representa la fibra original.
    :param num_extension_points: Número de puntos a añadir en cada extremo.
    :return: Array extendido de forma (25, 3).
    """
    if fiber.shape[0] != 21 or fiber.shape[1] != 3:
        raise ValueError("La fibra debe tener forma (21, 3)")

    
    direction_start = fiber[1] - fiber[0]

    # Calcular el vector dirección para la extensión en el extremo final
    direction_end = fiber[-1] - fiber[-2]

    # Generar puntos extendidos al inicio
    extended_start = [fiber[0] - direction_start * (i + 1) for i in range(num_extension_points)]
    
    extended_start = np.array(extended_start[::-1])  # Ordenar en el mismo sentido que la fibra

    # Generar puntos extendidos al final
    extended_end = [fiber[-1] + direction_end * (i + 1) for i in range(num_extension_points)]
    
    extended_end = np.array(extended_end)

    # Combinar los puntos extendidos con la fibra original
    extended_fiber = np.vstack([extended_start, fiber, extended_end])
    return extended_fiber

#%%
import trimesh
import numpy as np

def smooth_vertices(vertices, faces, iterations=10, alpha=0.5):
    """
    Suaviza los vértices de un mallado utilizando suavizado Laplaciano.
    :param vertices: Array (N, 3) con las coordenadas de los vértices.
    :param faces: Array (M, 3) con las caras (índices de vértices).
    :param iterations: Número de iteraciones de suavizado.
    :param alpha: Factor de suavizado (0.0 a 1.0).
    :return: Array (N, 3) con los vértices suavizados.
    """
    # Crear lista de vecinos para cada vértice
    neighbors = [[] for _ in tqdm(range(len(vertices)))]
    for face in faces:
        for i in range(3):
            neighbors[face[i]].extend(face[np.arange(3) != i])
    neighbors = [np.unique(neigh) for neigh in neighbors]

    # Aplicar suavizado iterativo
    smoothed_vertices = vertices.copy()
    for _ in tqdm(range(iterations)):
        new_vertices = smoothed_vertices.copy()
        for i, neigh in enumerate(neighbors):
            if len(neigh) > 0:
                new_vertices[i] = (1 - alpha) * smoothed_vertices[i] + alpha * smoothed_vertices[neigh].mean(axis=0)
        smoothed_vertices = new_vertices

    return smoothed_vertices

#%%
import trimesh
import numpy as np
from fury import window, actor
from tqdm import tqdm
def calculate_curvature_per_triangle(mesh, radius=0.1):
    """
    Calculate mean curvature for each triangle in a mesh.

    Parameters:
    - mesh: trimesh.Trimesh object.
    - radius: float, radius for curvature calculation.

    Returns:
    - triangle_curvature: numpy array of shape (M,), curvature for each triangle.
    """
    triangle_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.triangles_center, radius)
    return triangle_curvature

def map_triangle_curvature_to_vertices(vertices, faces, triangle_curvature):
    """
    Map triangle curvature to vertices by averaging over adjacent triangles.

    Parameters:
    - vertices: numpy array of shape (N, 3), vertices of the mesh.
    - faces: numpy array of shape (M, 3), faces of the mesh.
    - triangle_curvature: numpy array of shape (M,), curvature of each triangle.

    Returns:
    - vertex_curvature: numpy array of shape (N,), curvature averaged at each vertex.
    """
    # Initialize curvature values for each vertex
    vertex_curvature = np.zeros(len(vertices))
    vertex_count = np.zeros(len(vertices))

    # Accumulate triangle curvature to vertices
    for i, face in tqdm(enumerate(faces)):
        for vertex in face:
            vertex_curvature[vertex] += triangle_curvature[i]
            vertex_count[vertex] += 1

    # Average curvature at each vertex
    vertex_curvature /= np.clip(vertex_count, 1, None)  # Avoid division by zero
    return vertex_curvature

def classify_and_colorize(vertices, vertex_curvature, threshold=0):
    """
    Classify vertices into sulci and gyri based on curvature and assign colors.

    Parameters:
    - vertices: numpy array of shape (N, 3), vertices of the mesh.
    - vertex_curvature: numpy array of shape (N,), curvature averaged at each vertex.
    - threshold: float, curvature threshold to classify sulci and gyri.

    Returns:
    - colors: numpy array of shape (N, 3), RGB colors for each vertex.
    """
    sulci = vertex_curvature < threshold  # Curvature below threshold is sulci
    gyri = vertex_curvature >= threshold  # Curvature above threshold is gyri

    # Assign colors: red for sulci, green for gyri
    colors = np.zeros((len(vertices), 3))
    colors[sulci] = [1, 0, 0]  # Red for sulci
    colors[gyri] = [0, 1, 0]   # Green for gyri
    return colors

# Visualizar con Fury
def visualize_colored_mesh(vertices, faces, colors):
    """
    Visualize a mesh with colors using Fury.
    """
    scene = window.Scene()
    mesh_actor = actor.surface(vertices, faces, colors=colors)
    scene.add(mesh_actor)
    window.show(scene, title="Sulci and Gyri Coloring")

# Main execution
if __name__ == "__main__":
    # Example vertices and faces (replace with your actual data)
    vertices, faces = vertex_lh, polygons_lh

    # Crear malla
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Calcular curvatura por triángulo
    radius = 0.5  # Ajusta el radio según el tamaño de tu modelo
    triangle_curvature = calculate_curvature_per_triangle(mesh, radius)

    # Mapear curvatura a los vértices
    vertex_curvature = map_triangle_curvature_to_vertices(vertices, faces, triangle_curvature)

    # Clasificar y colorear los vértices
    colors = classify_and_colorize(vertices, vertex_curvature)
    visualize_colored_mesh(vertices, faces, colors)
    
#%%
import numpy as np

#%%
from scipy.ndimage import gaussian_filter
def norm(array):
    """Normaliza un array entre 0 y 1."""
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def compute_mean_curvature(vertices, faces):
    """
    Calcula la curvatura media aproximada para los vértices.
    """
    mean_curvature = np.zeros(len(vertices))
    area_accum = np.zeros(len(vertices))

    for f in faces:
        v1, v2, v3 = vertices[f]
        
        # Longitudes de los lados
        l1 = np.linalg.norm(v2 - v3)
        l2 = np.linalg.norm(v3 - v1)
        l3 = np.linalg.norm(v1 - v2)
        
        # Semiperímetro y área del triángulo
        s = (l1 + l2 + l3) / 2
        area = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))
        
        # Distribuir área entre los vértices
        area_accum[f] += area / 3
        
        # Curvatura por ángulos internos
        angles = np.array([
            np.arccos(np.dot(v2 - v1, v3 - v1) / (np.linalg.norm(v2 - v1) * np.linalg.norm(v3 - v1))),
            np.arccos(np.dot(v3 - v2, v1 - v2) / (np.linalg.norm(v3 - v2) * np.linalg.norm(v1 - v2))),
            np.arccos(np.dot(v1 - v3, v2 - v3) / (np.linalg.norm(v1 - v3) * np.linalg.norm(v2 - v3))),
        ])
        mean_curvature[f] += np.pi - angles  # Suma de ángulos "faltantes"
    
    # Normalizar curvatura por área acumulada
    mean_curvature /= area_accum
    return mean_curvature

def amplify_and_smooth_sulci(curvatures, sigma=1):
    """
    Amplifica y suaviza los surcos (valores negativos de curvatura).
    
    Parameters:
        curvatures (array): Curvaturas de los vértices.
        sigma (float): Parámetro para el suavizado gaussiano.
    
    Returns:
        array: Curvaturas amplificadas y suavizadas.
    """
    # Amplificar surcos (valores negativos)
    amplified = np.where(curvatures < 0, curvatures * 10, curvatures / 2)
    
    # Suavizado gaussiano para extender los surcos
    smoothed = gaussian_filter(amplified, sigma=sigma)
    
    return smoothed

curvatures = compute_mean_curvature(vertices_centered(smoothed_vertices), faces)

amplified_curvatures = amplify_and_smooth_sulci(norm(curvatures)-0.5, 1)

curvatures_normalized = norm(amplified_curvatures)


# Crear un mapa de colores: amarillo para giros, azul para surcos
colors = np.zeros((len(vertices), 3))
colors[:, 0] = norm(curvatures_normalized)  # Amarillo en giros
colors[:, 1] = norm(curvatures_normalized)  # Amarillo en giros
colors[:, 2] = 1 - norm(curvatures_normalized)+0.8  # Azul en surcos

colors

# colores_vertex = color_by_curvature(vertices, faces)
# colores_vertex
#%%

import trimesh
import numpy as np
import networkx as nx

def my_discrete_mean_curvature_measure(mesh):
    """Calculate discrete mean curvature of mesh using one-ring neighborhood."""

    # one-rings (immediate neighbors of) each vertex
    g = nx.from_edgelist(mesh.edges_unique)
    one_rings = [list(g[i].keys()) for i in range(len(mesh.vertices))]
   
    # cotangents of angles and store in dictionary based on corresponding vertex and face
    face_angles = mesh.face_angles_sparse
    cotangents = { f"{vertex},{face}":1/np.tan(angle) for vertex,face,angle in zip(face_angles.row, face_angles.col, face_angles.data)}

    # discrete Laplace-Beltrami contribution of the shared edge of adjacent faces:
    #        /*\
    #       / * \
    #      /  *  \
    #    vi___*___vj
    #
    # store results in dictionary with vertex ids as keys     
    fa = mesh.face_adjacency
    fae = mesh.face_adjacency_edges
    edge_measure = {f"{fae[i][0]},{fae[i][1]}":(mesh.vertices[fae[i][1]] - mesh.vertices[fae[i][0]])*(cotangents[f"{v[0]},{fa[i][0]}"]+cotangents[f"{v[1]},{fa[i][1]}"]) for i,v in enumerate(mesh.face_adjacency_unshared) }
  
    # calculate mean curvature using one-ring
    mean_curv = [0]*len(mesh.vertices)
    for vertex_id,face_ids in enumerate(mesh.vertex_faces):
        face_ids = face_ids[face_ids!=-1] #faces associated with vertex_id
        one_ring = one_rings[vertex_id]
        delta_s = 0;

        for one_ring_vertex_id in one_ring:
            if f"{vertex_id},{one_ring_vertex_id}" in edge_measure:
                delta_s += edge_measure[f"{vertex_id},{one_ring_vertex_id}"]
            elif f"{one_ring_vertex_id},{vertex_id}"  in edge_measure:
                delta_s -= edge_measure[f"{one_ring_vertex_id},{vertex_id}"]
        
        delta_s *= 1/(2*sum(mesh.area_faces[face_ids])/3) #use 1/3 of the areas
        mean_curv[vertex_id] = 0.5*np.linalg.norm(delta_s)
       
    return np.array(mean_curv)



# test on sphere of radius 5
radius = 0.1
m = mesh
discrete_mean_curvature = my_discrete_mean_curvature_measure(m)
print(discrete_mean_curvature)

from fury import window, actor
# Crear primera escena para el mallado original
colors = np.zeros((len(m.vertices), 3))
colors[:, 0] = norm(discrete_mean_curvature)  # Amarillo en giros
colors[:, 1] = norm(discrete_mean_curvature)  # Amarillo en giros
colors[:, 2] = 1 - norm(discrete_mean_curvature) # Azul en surcos
scene_original = window.Scene()
original_actor = actor.surface(m.vertices, m.faces, colors) # Azul para el mallado original
scene_original.add(original_actor)
window.show(scene_original)

    

#%%
from scipy.spatial import KDTree
import numpy as np

point0 = centered_fibers[4][10]
kdtree = KDTree(centered_vertices)

# Centroide en ambos espacios
centroid = centered_vertices.mean(axis=0)
spherical_centroid = centered_spherical_vertices.mean(axis=0)

# Encontrar el vértice más cercano
_, indices = kdtree.query(point0)
nearest_vertice = centered_vertices[indices]
nearest_spherical_vertice = centered_spherical_vertices[indices]

# Relación triangular en el espacio cartesiano
A = np.linalg.norm(point0 - nearest_vertice)
B = np.linalg.norm(centroid - nearest_vertice)
C = np.linalg.norm(point0 - centroid)

# Relación triangular en el espacio esférico
B_p = np.linalg.norm(nearest_spherical_vertice - spherical_centroid)
A_p = A * B_p / B
C_p = C * B_p / B

# Direcciones normalizadas
nA = (point0 - nearest_vertice) / A if A != 0 else np.zeros_like(point0)
nC = (point0 - centroid) / C if C != 0 else np.zeros_like(point0)

print(np.linalg.norm(nA), np.linalg.norm(nC))
# Nuevas posiciones en el espacio esférico
point0_pA = centered_spherical_vertices[indices] + A_p * nA
point0_pC = spherical_centroid + C_p * nC

print(point0_pA, point0_pC)
#%%

scene = window.Scene()
actor_vertices = actor.surface(centered_vertices, faces)
actor_vertices.GetProperty().SetOpacity(0.5)
scene.add(actor_vertices)

actor_points = actor.point([centroid, nearest_vertice, point0], colors = (1,0,0), point_radius=.01)
scene.add(actor_points)

actor_points = actor.point(centered_fibers[41], colors = (1,1,0), point_radius=.01)
scene.add(actor_points)

actor_fiber = actor.line([centered_fibers[30]], linewidth=2)
scene.add(actor_fiber)

window.show(scene)
#%%
scene = window.Scene()
actor_vertices = actor.surface(centered_spherical_vertices, faces)
actor_vertices.GetProperty().SetOpacity(0.2)
scene.add(actor_vertices)

actor_points = actor.point([spherical_centroid, nearest_spherical_vertice, point0_pA], colors = (1,0,0), point_radius=.01)
scene.add(actor_points)

# actor_points = actor.point(rbf_points, colors = (1,1,0), point_radius=.01)
# scene.add(actor_points)

window.show(scene)
#%%
from scipy.spatial import KDTree
import numpy as np

bundle_sphere = project_points_with_nearest_neighbor(centered_vertices, centered_spherical_vertices, bundle[12])
#%%

import numpy as np

from scipy.spatial import KDTree

def map_fibers_with_interpolation(vertices_cartesian, vertices_spherical, fibers, center=None):
    """
    Mapea fibras al espacio esférico utilizando interpolación de vecinos cercanos.
    
    Parameters:
        vertices_cartesian (numpy array): Vértices de la malla cortical en espacio cartesiano (n, 3).
        vertices_spherical (numpy array): Vértices de la malla cortical en espacio esférico (n, 3).
        fibers (numpy array): Fibras en espacio cartesiano (m, p, 3).
    
    Returns:
        numpy array: Fibras mapeadas al espacio esférico (m, p, 3).
    """
    # Crear KD-Tree para encontrar vecinos cercanos
    kdtree = KDTree(vertices_cartesian)
    
    fibers_spherical = []
    for fiber in fibers:
        fiber_spherical = []
        for point in fiber:
            # Encontrar el vértice más cercano
            dist, index = kdtree.query(point)
            # Calcular la dirección relativa al vértice más cercano
            direction = point - vertices_cartesian[index]
            # Mover el punto en el espacio esférico usando la misma dirección
            mapped_point = vertices_spherical[index] + direction
            fiber_spherical.append(mapped_point)
        fibers_spherical.append(fiber_spherical)
    
    return np.array(fibers_spherical)


centered_vertices = (vertices-vertices.mean(axis = 0))
vertices_norm_max = np.linalg.norm(centered_vertices, axis = 1).max()
print(vertices_norm_max)
centered_vertices /= vertices_norm_max
centered_fibers = (bundle[:10000]-vertices.mean(axis = 0))/vertices_norm_max
centered_spherical_vertices = spherical_vertices-spherical_vertices.mean(axis = 0)
tranformed_fibers = map_fibers_with_interpolation(centered_vertices, centered_spherical_vertices, centered_fibers, 1)

#%%
from fury import actor, window
scene = window.Scene()
actor_vertices = actor.surface(centered_vertices, faces)
actor_vertices.GetProperty().SetOpacity(0.5)
# scene.add(actor_vertices)
actor_bundle = actor.point(bundle_sphere, colors = (0,1,0), point_radius=.01)
scene.add(actor_bundle)
actor_spherical_vertices = actor.surface(centered_spherical_vertices, faces, colors = np.array([[0,0,1]]*len(vertices)))
actor_spherical_vertices.GetProperty().SetOpacity(0.5)
scene.add(actor_spherical_vertices)

# actor_bundle = actor.line(tranformed_fibers)
# scene.add(actor_bundle)
window.show(scene)
#%%
n_fibers = 20
fiber_points = np.reshape(bundle[:n_fibers],(-1, 3))
transformed_points = transform_large_set(build_kdtree(vertices), vertices, spherical_vertices, fiber_points, k = 10)
transformed_points = transformed_points.reshape(n_fibers, 21, 3)
#%%
from scipy.interpolate import Rbf
import numpy as np

def build_rbf_interpolator(source_points, target_points, num_samples=10000):
    """
    Construye un interpolador RBF utilizando submuestreo de los puntos de referencia.
    
    Parameters:
        source_points (numpy array): Coordenadas de los puntos en el espacio original (n, 3).
        target_points (numpy array): Coordenadas de los puntos en el espacio transformado (n, 3).
        num_samples (int): Número de puntos de referencia a usar para el submuestreo.
    
    Returns:
        tuple: Interpoladores RBF para cada dimensión (x, y, z).
    """
    # Submuestreo de puntos de referencia
    indices = np.random.choice(len(source_points), size=num_samples, replace=False)
    sampled_source = source_points[indices]
    sampled_target = target_points[indices]
    
    # Crear interpoladores para cada dimensión
    rbf_x = Rbf(sampled_source[:, 0], sampled_source[:, 1], sampled_source[:, 2], sampled_target[:, 0], function='multiquadric')
    rbf_y = Rbf(sampled_source[:, 0], sampled_source[:, 1], sampled_source[:, 2], sampled_target[:, 1], function='multiquadric')
    rbf_z = Rbf(sampled_source[:, 0], sampled_source[:, 1], sampled_source[:, 2], sampled_target[:, 2], function='multiquadric')
    
    return rbf_x, rbf_y, rbf_z

def transform_points_with_rbf(rbf_x, rbf_y, rbf_z, points_to_transform, batch_size=10000):
    """
    Transforma un gran conjunto de puntos utilizando interpoladores RBF en lotes.
    
    Parameters:
        rbf_x, rbf_y, rbf_z: Interpoladores RBF para cada dimensión.
        points_to_transform (numpy array): Puntos a transformar (m, 3).
        batch_size (int): Tamaño del lote para procesar.
    
    Returns:
        numpy array: Puntos transformados (m, 3).
    """
    n_points = len(points_to_transform)
    transformed_points = np.zeros_like(points_to_transform)
    
    for i in range(0, n_points, batch_size):
        batch = points_to_transform[i:i + batch_size]
        transformed_x = rbf_x(batch[:, 0], batch[:, 1], batch[:, 2])
        transformed_y = rbf_y(batch[:, 0], batch[:, 1], batch[:, 2])
        transformed_z = rbf_z(batch[:, 0], batch[:, 1], batch[:, 2])
        transformed_points[i:i + batch_size] = np.column_stack((transformed_x, transformed_y, transformed_z))
    
    return transformed_points

n_fibers = 100
fiber_points = np.reshape(bundle[:n_fibers],(-1, 3))
transformed_points = rbf_transform(vertices, spherical_vertices, fiber_points)

#%%
from fury import window, actor
# Crear primera escena para el mallado original
scene_original = window.Scene()
original_actor = actor.surface(spherical_vertices, faces) # Azul para el mallado original

# scene_original.add(original_actor)

original_actor = actor.surface(centered_spherical_vertices, faces) # Azul para el mallado original
original_actor.GetProperty().SetOpacity(0.5)
scene_original.add(original_actor)


# points_actor = actor.point(bundle_sphere, colors = (1,0,0), point_radius=0.01)
# scene_original.add(points_actor)

# points_actor = actor.point(spherical_vertices[:1000], colors = (0,1,0), point_radius=0.08)
# scene_original.add(points_actor)

window.show(scene_original, title="Mallado Original")
#%%

from fury import window, actor
# Crear primera escena para el mallado original
scene_original = window.Scene()
original_actor = actor.surface(spherical_vertices, faces) # Azul para el mallado original
original_actor.GetProperty().SetOpacity(0.2)
scene_original.add(original_actor)
fibers_actor = actor.line(tranformed_fibers, linewidth=1)
scene_original.add(fibers_actor)
window.show(scene_original, title="Mallado Original")

#%%
scene_original = window.Scene()
original_actor = actor.surface(spherical_vertices, faces, colors = colors) # Azul para el mallado original
original_actor.GetProperty().SetOpacity(0.2)
scene_original.add(original_actor)
fibers_actor = actor.line(bundle_sphere[:10], linewidth=1)
scene_original.add(fibers_actor)
window.show(scene_original, title="Mallado Original")

#%%
from fury import window, actor
# Crear primera escena para el mallado original
norm_smoothed_vertices = smoothed_vertices
scene_original = window.Scene()
original_actor = actor.surface(norm_smoothed_vertices, polygons_lh, colors = colors) # Azul para el mallado original
original_actor.GetProperty().SetOpacity(0.8)
scene_original.add(original_actor)

fibers_actor = actor.line(np.array(bundle[:10]) - vertices.mean(axis = 0))
scene_original.add(fibers_actor)

fibers_spherical_actor = actor.line(bundle_sphere[:10])
# scene_original.add(fibers_spherical_actor)

inflated_actor = actor.surface(spherical_vertices, polygons_lh, colors = colors)  # Rojo para el mallado inflado
# inflated_actor.GetProperty().SetInterpolationToPhong() 
# scene_original.add(inflated_actor)

point_smooth_actor = actor.point([spherical_vertices[1]], colors = (0,0,1), point_radius=0.03)
# scene_original.add(point_smooth_actor)

point_spherical_actor = actor.point([1+norm_smoothed_vertices[12]], colors = (1,0,1), point_radius=.03)
scene_original.add(point_spherical_actor)
# Mostrar ambas escenas en ventanas separadas
window.show(scene_original, title="Mallado Original")

#%%
normalize = (vertices[:, 2] - vertices[:, 2].min())/(vertices[:, 2].max() - vertices[:, 2].min())
[[1,0,0]*len(normalize)]
#%%
from fury import window, actor
scene = window.Scene()

# Actor para el mallado original
original_actor = actor.surface(vertex_lh, polygons_lh, colors=np.array([[1,0,0]]*len(vertex_lh)))  # Azul para el mallado original
original_actor.GetProperty().SetOpacity(.8)
scene.add(original_actor)

# Actor para el mallado original
wraped_actor = actor.surface(vertex_lh, hull_faces)  # Azul para el mallado original
# wraped_actor.GetProperty().SetOpacity(.98)
# scene.add(wraped_actor)

point_actor = actor.point([centroid], colors = (0,1,0), point_radius=5)
scene.add(point_actor)

# Mostrar escena
window.show(scene)

#%%
from scipy.spatial import ConvexHull
convex_hull = ConvexHull(vertex_lh)
vertex_hull = vertex_lh[convex_hull.vertices]
hull_faces = convex_hull.simplices
