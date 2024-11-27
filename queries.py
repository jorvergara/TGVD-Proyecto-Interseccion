# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:53:00 2024

@author: Jorge
"""

import numpy as np
from tqdm import tqdm
from intersection_rays_utils import ray_triangle_intersection_vector, ray_triangle_intersection_matrix

def compute_directions(bundle):
    bundle = np.array(bundle)
    direction = np.zeros_like(bundle)

    for i in range(bundle.shape[0]):
        for j in range(bundle.shape[1]):
            if(j == bundle.shape[1]-1):
                direction[i,j] = -(bundle[i,-2] - bundle[i,-1])
                
            elif(j == 0):
                direction[i,j] = bundle[i,1] - bundle[i,0]
            else:
                direction[i,j] = bundle[i,j+1] - bundle[i,j]
    direction = np.concat([-direction[:,0][:,np.newaxis], direction], axis = 1)
    bundle = np.concat([bundle[:,0][:,np.newaxis], bundle], axis = 1)
    
    return bundle, direction

def query_intersection_brute(n_fibers, bundle, vertices, faces, seed = None):
    bundle = np.array(bundle)
    vertices = np.array(vertices)
    faces = np.array(faces)
    """
    Realiza un cálculo de intersección fuerza bruta entre fibras y un conjunto de triángulos.
    
    Args:
        n_fibers (int): Número de fibras a seleccionar aleatoriamente del bundle.
        bundle (np.ndarray): Array de fibras de forma (num_fibers, num_points, 3).
        vertices (np.ndarray): Array de vértices de los triángulos (num_vertices, 3).
        faces (np.ndarray): Array de índices de los triángulos (num_faces, 3).
    
    Returns:
        list: Resultados de intersecciones para las fibras seleccionadas.
    """
    # Configura la semilla si se proporciona
    if seed is not None:
        np.random.seed(seed)
        
    # Seleccionar fibras aleatoriamente
    total_fibers = bundle.shape[0]
    if n_fibers > total_fibers:
        raise ValueError(f"n_fibers ({n_fibers}) es mayor que el total de fibras disponibles ({total_fibers}).")
    
    # Índices aleatorios
    random_indices = np.random.choice(total_fibers, n_fibers, replace=False)
    selected_fibers = bundle[random_indices]
    
    origen, direction = compute_directions(selected_fibers)
    
    intersection = []
    for i in tqdm(range(n_fibers)):
        intersection_per_fiber = ray_triangle_intersection_matrix(origen[i], direction[i], vertices[faces]).sum(axis = 1)
        intersection_per_fiber = np.where(intersection_per_fiber)[0].tolist()
        intersection.append(intersection_per_fiber)
    
    return selected_fibers, intersection

def query_intersection_octree(octree, n_fibers, bundle, vertices, faces, seed = None):
    bundle = np.array(bundle)
    vertices = np.array(vertices)
    faces = np.array(faces)
    """
    Realiza un cálculo de intersección fuerza bruta entre fibras y un conjunto de triángulos.
    
    Args:
        n_fibers (int): Número de fibras a seleccionar aleatoriamente del bundle.
        bundle (np.ndarray): Array de fibras de forma (num_fibers, num_points, 3).
        vertices (np.ndarray): Array de vértices de los triángulos (num_vertices, 3).
        faces (np.ndarray): Array de índices de los triángulos (num_faces, 3).
    
    Returns:
        list: Resultados de intersecciones para las fibras seleccionadas.
    """
    # Configura la semilla si se proporciona
    if seed is not None:
        np.random.seed(seed)
        
    # Seleccionar fibras aleatoriamente
    total_fibers = bundle.shape[0]
    if n_fibers > total_fibers:
        raise ValueError(f"n_fibers ({n_fibers}) es mayor que el total de fibras disponibles ({total_fibers}).")
    
    # Índices aleatorios
    random_indices = np.random.choice(total_fibers, n_fibers, replace=False)
    selected_fibers = bundle[random_indices]
    
    origen, direction = compute_directions(selected_fibers)
    
    adjacent_triangles = []
    for i in tqdm(range(origen.shape[0])):
        adjacent_per_fiber = []
        for j in range(origen.shape[1]):
            adjacent_per_fiber.append(octree.get_adjacent_triangles(origen[i,j]))
        adjacent_triangles.append(adjacent_per_fiber)
    
    intersection = []
    for i in tqdm(range(origen.shape[0])):
        adjacent_per_fiber = []
        for j in range(origen.shape[1]):
            if(adjacent_triangles[i][j].size > 0):
                intersection_per_point = np.array(ray_triangle_intersection_vector(origen[i,j], direction[i,j], vertices[faces[adjacent_triangles[i][j]]])).flatten()
                triangles_intersected = adjacent_triangles[i][j][intersection_per_point].tolist()
                adjacent_per_fiber.extend(triangles_intersected)
        intersection.append(list(set(adjacent_per_fiber)))
        
    return selected_fibers, adjacent_triangles, intersection