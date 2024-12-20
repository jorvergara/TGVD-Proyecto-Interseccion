# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:53:00 2024

@author: Jorge
"""
import sys
sys.path.append('build')
import numpy as np
from tqdm import tqdm

import octree_module


def compute_directions(bundle):
    """
    Calcula las direcciones de las fibras en un bundle.
    """
    bundle = np.array(bundle)
    origin = bundle[:,:-1]
    direction = np.diff(bundle, axis=1)    
    return origin, direction

def query_intersection_brute(n_fibers, bundle, vertices, faces, seed = None):
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
    bundle = np.array(bundle)
    vertices = np.array(vertices)
    faces = np.array(faces)

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
    selected_fibers = np.array([proyection_fiber(fiber) for fiber in selected_fibers])
    
    origen, direction = compute_directions(selected_fibers)
    
    intersection = []
    intersection = octree_module.ray_triangle_intersection(origen, direction, vertices, faces)
    
    return random_indices, intersection

def proyection_fiber(fiber, tolerance = 5, vox_resolution = 0.7):
    tolerance_vox = tolerance / vox_resolution

    origin, direction = fiber[0], fiber[0] - fiber[1]
    direction = direction / np.linalg.norm(direction)
    fiber_proyected = np.zeros((len(fiber) + 2, 3))
    fiber_proyected[0] = origin + direction * tolerance_vox
    fiber_proyected[1:-1] = fiber

    origin, direction = fiber[-1], fiber[-1] - fiber[-2]
    direction = direction / np.linalg.norm(direction)
    fiber_proyected[-1] = origin + direction * tolerance_vox

    return fiber_proyected
