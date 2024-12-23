import numpy as np
import pickle
import intersection_utils as iu
from utilities import obtener_indices_por_clase

def gen_pares_intersecciones(lista_intersecciones):
    return {
        (fibra_idx, tri_idx) 
        for fibra_idx, lista_triangulos in enumerate(lista_intersecciones) 
        for tri_idx in lista_triangulos
    }

def pair_recall(set_a, set_b):
    if not set_b:
        return 0
    return len(set_a & set_b) / len(set_b)

def pair_discrepancy(set_a, set_b):
    return len(set_a - set_b) + len(set_b - set_a)

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

# Función para calcular métricas de comparación
def calculate_metrics(N):
    inclusion_metrics = []
    discrepancy_metrics = []
    intersections_felipe = []
    intersection_octree = []

    for i in N:
        # Cargar datos de Felipe
        results_path = f'results/test_{i}/'
        intersection_file = results_path + f'3Msift_t_MNI_21p_bundle_{i}.intersectiondata'
        InTri, FnTri, _, _, fib_index, fiber_class = iu.read_intersection(intersection_file)

        triangulos_felipe, fibras_felipe = obtener_indices_por_clase(InTri, FnTri, fib_index, fiber_class)
        intersection_pares_felipe = {(fibra, triangulo) for fibra, triangulo in zip(fibras_felipe, triangulos_felipe)}

        # Cargar datos del Octree
        result_path_octree = f'results/test_octree{i}/'
        intersection_file_octree = result_path_octree + f'intersection_octree_{i}.pkl'
        with open(intersection_file_octree, "rb") as f:
            insersection_octree = pickle.load(f)
        intersection_pares_octree = gen_pares_intersecciones(insersection_octree)

        # Calcular métricas
        inclusion_metric_felipe = inclusion_metric(intersection_pares_felipe, intersection_pares_octree)
        discrepancy_metric_felipe = discrepancy_metric(intersection_pares_felipe, intersection_pares_octree)

        inclusion_metrics.append(inclusion_metric_felipe)
        discrepancy_metrics.append(discrepancy_metric_felipe)
        intersections_felipe.append(len(intersection_pares_felipe))
        intersection_octree.append(len(intersection_pares_octree))

    return inclusion_metrics, discrepancy_metrics, intersections_felipe, intersection_octree