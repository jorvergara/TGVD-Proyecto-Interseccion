# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:49:29 2024

@author: User
"""

import numpy as np

class Triangle:
    def __init__(self, vertices, indices, idx):
        """
        Inicializa un triángulo.
        :param vertices: Lista de coordenadas de los vértices (3x3 array).
        :param indices: Índices de los vértices en el arreglo original.
        """
        self.vertices = np.array(vertices)  # Coordenadas de los vértices (3x3 array)
        self.indices = indices              # Índices de los vértices en el arreglo original
        self.normal = self.calculate_normal()  # Vector normal, calculado al inicializar
        self.area = self.calculate_area()
        self.idx = idx
        
    def calculate_normal(self):
        """
        Calcula el vector normal de un triángulo.
        :return: Vector normal unitario (normalizado).
        """
        A, B, C = self.vertices
        u = B - A
        v = C - A
        normal = np.cross(u, v)
        magnitude = np.linalg.norm(normal)
        if magnitude == 0:  # Evitar división por cero (triángulo degenerado)
            return np.array([0, 0, 0])
        return normal / magnitude
    
    def calculate_area(self):
        """
        Calcula el área de un triángulo.
        :param triangle: Objeto Triangle.
        :return: Área del triángulo.
        """
        A, B, C = self.vertices
        return 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    
    def __repr__(self):
        """
        Representación legible del triángulo.
        """
        return f"Triangle(vertices={self.vertices.tolist()}, indices={self.indices}, normal={self.normal.tolist()}, area= {self.area.tolist()}, indice = {self.idx})"
    
class OctreeNode:
    def __init__(self, bounds, depth=0, min_depth = 5, max_depth=7, epsilon = 0.01):
        """
        Nodo básico del Octree.
        :param bounds: [xmin, ymin, zmin, xmax, ymax, zmax]
        :param depth: Profundidad actual del nodo.
        :param max_depth: Profundidad máxima del Octree.
        :param threshold: Cantidad máxima de triángulos antes de subdividir el nodo.
        """
        self.bounds = bounds
        self.children = []  # Hijos del nodo
        self.triangles = []  # Lista de objetos Triangle
        self.depth = depth
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.epsilon = epsilon
        
    def insert_scene(self, scene):
        for triangle in scene:
            self.triangles.append(triangle)

    def subdivide(self):
        """Divide este nodo en 8 hijos."""
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounds
        xmid, ymid, zmid = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        self.children = [
            OctreeNode([xmin, ymin, zmin, xmid, ymid, zmid], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmid, ymin, zmin, xmax, ymid, zmid], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmin, ymid, zmin, xmid, ymax, zmid], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmid, ymid, zmin, xmax, ymax, zmid], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmin, ymin, zmid, xmid, ymid, zmax], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmid, ymin, zmid, xmax, ymid, zmax], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmin, ymid, zmid, xmid, ymax, zmax], self.depth + 1, self.min_depth, self.max_depth, self.epsilon),
            OctreeNode([xmid, ymid, zmid, xmax, ymax, zmax], self.depth + 1, self.min_depth, self.max_depth, self.epsilon)
        ]
    
    def calculate_normal_variance(self):
        """
        Calcula la métrica L_r^{2,1} para este nodo del Octree.
        :param epsilon: Tolerancia para la subdivisión.
        :return: True si se debe subdividir el nodo, False en caso contrario.
        """
        if not self.triangles:
            return False  # No hay triángulos en el nodo, no subdividir

        # Calcular normales y áreas
        normals = []
        areas = []
        for triangle in self.triangles:
            normals.append(triangle.normal)
            areas.append(triangle.area)

        normals = np.array(normals)
        areas = np.array(areas)

        # Calcular el promedio ponderado de las normales
        avg_normal = np.sum(normals.T * areas, axis=1) / np.sum(areas)
        avg_normal /= np.linalg.norm(avg_normal)  # Normalizar el promedio para obtener un vector unitario

        # Calcular L_r^{2,1}
        variance = np.sum(np.linalg.norm(normals - avg_normal, axis=1) ** 2)
        
        return variance > self.epsilon
    
    def insert_triangle(self, triangle):
        """
        Inserta un triángulo en este nodo o en sus hijos.
        :param triangle: Objeto de la clase Triangle.
        """
        # if self.depth >= self.max_depth:
        #     self.triangles.append(triangle)
        #     return
        
        p1 = len(self.triangles) == 0
        p2 = self.depth >= self.max_depth 
        p3 = self.depth >= self.min_depth and not self.calculate_normal_variance()
        
        # Si no tiene hijos, subdividir si excede el umbral
        if p1 or p2 or p3:
            pass
        else:
            self.subdivide()

            # Redistribuir triángulos existentes
            while self.triangles:
                existing_triangle = self.triangles.pop()
                for child in self.children:
                    if is_triangle_in_bounds(existing_triangle.vertices, child.bounds):
                        child.insert_triangle(existing_triangle)

        # Si el nodo aún no tiene hijos
        if not self.children:
            self.triangles.append(triangle)
            return

        # Insertar en los hijos correspondientes
        for child in self.children:
            if is_triangle_in_bounds(triangle.vertices, child.bounds):
                child.insert_triangle(triangle)
                
    def recursive_subdivision(self):
        """
        Realiza una subdivisión recursiva del Octree basada en los criterios descritos.
        :param min_depth: Profundidad mínima antes de aplicar el criterio de variación de normales.
        :param max_depth: Profundidad máxima de subdivisión.
        :param epsilon: Tolerancia para la variación de normales.
        """
        # Criterios de terminación
        
        if len(self.triangles) == 0 or len(self.triangles) <20:  # El nodo no contiene triángulos
            return

        if self.depth >= self.max_depth:  # Se alcanzó la profundidad máxima
            return

        if self.depth >= self.min_depth and not self.calculate_normal_variance():
            # La variación de normales es menor que la tolerancia
            return

        # Subdividir el nodo actual
        self.subdivide()

        # Redistribuir los triángulos entre los hijos
        while self.triangles:
            triangle = self.triangles.pop()
            for child in self.children:
                if is_triangle_in_bounds(triangle.vertices, child.bounds):
                    child.triangles.append(triangle)

        # Llamada recursiva para cada hijo
        for child in self.children:
            child.recursive_subdivision()
    
    def find_leaf_containing_point(self, point):
        """
        Encuentra el nodo hoja que contiene el punto.
        :param point: Coordenadas del punto [x, y, z].
        :return: Nodo hoja que contiene el punto o None si no está dentro de los límites del Octree.
        """
        # Verificar si el punto está dentro de los límites del nodo actual.
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounds
        if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and zmin <= point[2] <= zmax):
            return None  # El punto está fuera de los límites.

        # Si no tiene hijos, este es un nodo hoja.
        if not self.children:
            return self

        # Si tiene hijos, buscar en ellos.
        for child in self.children:
            result = child.find_leaf_containing_point(point)
            if result:
                return result

        return None  # No se encontró en ninguno de los hijos.
    
    def get_adjacent_triangles(self, point):
        """
        Encuentra todos los triángulos adyacentes al nodo que contiene el punto dado.
        :param point: Coordenadas del punto [x, y, z].
        :return: Array de triángulos adyacentes únicos.
        """
        # Encontrar el nodo hoja que contiene el punto
        leaf_node = self.find_leaf_containing_point(point)
        if not leaf_node:
            return np.array([])  # Retornar vacío si no se encuentra el nodo
    
        # Obtener los límites y el punto central del nodo
        x0, y0, z0, x1, y1, z1 = leaf_node.bounds
        central_point = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    
        # Calcular los puntos adyacentes
        directions = [-1, 0, 1]
        adjacent_points = [
            (central_point[0] + dx * (x1 - x0),
             central_point[1] + dy * (y1 - y0),
             central_point[2] + dz * (z1 - z0))
            for dx in directions
            for dy in directions
            for dz in directions
            if not (dx == 0 and dy == 0 and dz == 0)  # Excluir el punto central
        ]
    
        # Encontrar los nodos adyacentes
        adjacent_nodes = set([self.find_leaf_containing_point(pt) for pt in adjacent_points])
        adjacent_nodes.add(leaf_node)  # Incluir el nodo actual
        adjacent_nodes.discard(None)  # Eliminar nodos nulos
    
        # Recopilar triángulos únicos
        triangles_adjacent = []
        for node in adjacent_nodes:
            triangles_adjacent.extend(node.triangles)
    
        # Eliminar duplicados y retornar como un array
        unique_triangles = set([triangle.idx for triangle in triangles_adjacent])
        return np.array(list(unique_triangles))

def is_triangle_in_bounds(triangle_vertices, bounds):
    """
    Verifica si un triángulo está dentro de los límites del nodo.
    :param triangle_vertices: Coordenadas de los vértices del triángulo (3x3 array).
    :param bounds: Límites del nodo [xmin, ymin, zmin, xmax, ymax, zmax].
    :return: True si al menos un vértice del triángulo está dentro de los límites, False de lo contrario.
    """
    for vertex in triangle_vertices:
        if (bounds[0] <= vertex[0] <= bounds[3] and
            bounds[1] <= vertex[1] <= bounds[4] and
            bounds[2] <= vertex[2] <= bounds[5]):
            return True
    return False

def print_octree_content(node, depth=0):
    """Imprime el contenido del Octree recursivamente."""
    indent = "  " * depth
    print(f"{indent}Nodo a profundidad {depth}: {len(node.triangles)} triángulos")
    for child in node.children:
        print_octree_content(child, depth + 1)

def count_triangles(node):
    """Suma los triángulos de un nodo y todos sus hijos recursivamente."""
    # Comienza con los triángulos del nodo actual
    total_triangles = len(node.triangles)
   
    # Agrega los triángulos de los hijos
    for child in node.children:
        total_triangles += count_triangles(child)
   
    return total_triangles