#%%
import sys
sys.path.append('build')
import timeit
import numpy as np
import matplotlib.pyplot as plt
import intersection_utils as iu
import octree_module

import numpy as np
import bundleTools as BT
import bundleTools3 as BT3
import visualizationTools as vt
from tqdm import tqdm
import pickle
import time
import os
from fury import window, actor

from utilities import *
from metrics import *

def evaluate_octree_performance():
    """
    Compara el rendimiento del Octree SAT y Octree Vertex en términos de construcción
    y cálculo de intersecciones.
    """
    # Cargar datos
    print("Cargando datos...")
    bundle, vertex_lh, polygons_lh = load_data()

    # Calcular límites
    bounds = np.concatenate([np.min(vertex_lh, axis=0), np.max(vertex_lh, axis=0)]).astype(np.float32)
    
    # Construir octree SAT
    print("Construyendo Octree SAT...")
    start_time = time.time()
    octree = build_octree(bounds, vertex_lh, polygons_lh)
    construction_time = time.time() - start_time
    print(f"Tiempo de construcción (Octree SAT): {construction_time:.4f}s")

    # Construir octree Vertex
    print("Construyendo Octree Vertex...")
    start_time = time.time()
    octree_vertex = build_octree_vertex(bounds, vertex_lh, polygons_lh)
    construction_time_vertex = time.time() - start_time
    print(f"Tiempo de construcción (Octree Vertex): {construction_time_vertex:.4f}s")

    # Graficar comparación de tiempos de construcción
    plt.figure(figsize=(10, 10))
    plt.bar(["Octree SAT", "Octree Vertex"], [construction_time, construction_time_vertex])
    plt.ylabel("Tiempo de Construcción (segundos)")
    plt.title("Comparación de Tiempos de Construcción")
    plt.grid(True)
    plt.savefig("figures/octree_construction_time.png", dpi=300)

    # Cargar o calcular intersecciones GT
    print("Calculando intersecciones GT...")
    selected_fibers_gt, intersection_gt = calculate_intersections_gt(1000, bundle, vertex_lh, polygons_lh)

    # Centrar datos para intersecciones
    centered_bundle = bundle - bounds[:3]
    centered_vertices = vertex_lh - bounds[:3]

    # Calcular intersecciones con Octree SAT
    print("Calculando intersecciones con Octree SAT...")
    intersection, _ = octree_module.find_intersections(
        octree, centered_bundle[selected_fibers_gt], centered_vertices, polygons_lh
    )

    # Calcular intersecciones con Octree Vertex
    print("Calculando intersecciones con Octree Vertex...")
    intersection_vertex, _ = octree_module.find_intersections(
        octree_vertex, centered_bundle[selected_fibers_gt], centered_vertices, polygons_lh
    )

    # Generar pares de intersecciones
    print("Calculando métricas...")
    pairs_gt = gen_pares_intersecciones(intersection_gt)
    pairs_sat = gen_pares_intersecciones(intersection)
    pairs_vertex = gen_pares_intersecciones(intersection_vertex)

    # Métricas para SAT
    recall_sat = pair_recall(pairs_sat, pairs_gt)
    discrepancy_sat = pair_discrepancy(pairs_sat, pairs_gt)

    # Métricas para Vertex
    recall_vertex = pair_recall(pairs_vertex, pairs_gt)
    discrepancy_vertex = pair_discrepancy(pairs_vertex, pairs_gt)

    print(f"Recall (Octree SAT): {recall_sat:.4f}")
    print(f"Recall (Octree Vertex): {recall_vertex:.4f}")
    print(f"Discrepancia (Octree SAT): {discrepancy_sat}")
    print(f"Discrepancia (Octree Vertex): {discrepancy_vertex}")

    # Guardar métricas en archivo
    with open("results/octree_metrics.txt", "w") as f:
        f.write(f"Recall (Octree SAT): {recall_sat:.4f}\n")
        f.write(f"Recall (Octree Vertex): {recall_vertex:.4f}\n")
        f.write(f"Discrepancia (Octree SAT): {discrepancy_sat}\n")
        f.write(f"Discrepancia (Octree Vertex): {discrepancy_vertex}\n")


def setup_environment(N):
    """
    Configura el entorno cargando los datos, creando los test bundles y construyendo el Octree.
    """
    print("Cargando datos...")
    bundle, vertex_lh, polygons_lh = load_data()

    print("Creando test bundles...")
    create_multiple_test_bundles(N, bundle)

    print("Construyendo Octree...")
    bounds = np.concatenate([np.min(vertex_lh, axis=0), np.max(vertex_lh, axis=0)]).astype(np.float32)
    octree = build_octree(bounds, vertex_lh, polygons_lh)

    return bundle, vertex_lh, polygons_lh, bounds, octree

def evaluate_algorithms(N, bundle, vertex_lh, polygons_lh, bounds, octree):
    """
    Evalúa los algoritmos de intersección con Felipe y Octree.
    """
    # Evaluar Felipe
    print("Evaluando Felipe...")
    times_felipe = test_felipe(N, 'meshes/')

    # Evaluar Octree
    print("Evaluando Octree...")
    times_octree = test_octree(N, bounds, vertex_lh, polygons_lh, octree)

    # Graficar comparación de tiempos de intersección
    plt.figure(figsize=(10, 10))
    plt.plot(N, times_felipe, "-o", label="Felipe")
    plt.plot(N, times_octree, "-o", label="Octree")
    plt.xlabel("Número de Fibras")
    plt.ylabel("Tiempo de Ejecución (segundos)")
    plt.title("Comparación de Tiempos de Ejecución")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/times_comparison.png", dpi=300)
    
    # Calcular métricas de comparación
    inclusion_metrics, discrepancy_metrics, intersections_felipe, intersection_octree = calculate_metrics(N)

    # Comparación de detección de intersecciones entre Felipe y Octree
    plt.figure(figsize=(10, 10))
    plt.plot(N, intersections_felipe, label="Total de intersecciones Felipe")
    plt.plot(N, intersection_octree, label="Total de intersecciones Octree")
    plt.xlabel("Número de Fibras")
    plt.ylabel("Número de Intersecciones")
    plt.title("Comparación de Intersecciones")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/intersections_comparison.png", dpi=300)

    # Comparación con dibras de felipe que no estan en el octree (fp)
    plt.figure(figsize=(10, 10))
    plt.plot(N, intersections_felipe, label="Total de intersecciones Felipe")
    plt.plot(N, discrepancy_metrics, label="Discrepancia")
    plt.xlabel("Número de Fibras")
    plt.ylabel("Número de Intersecciones")
    plt.title("Comparación de Intersecciones")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/discrepancia.png", dpi=300)
    
    # Fibras que comparten intersecciones en Felipe y Octree (tp)
    plt.figure(figsize=(10, 10))
    plt.plot(N, inclusion_metrics, label="Inclusión")
    plt.xlabel("Número de Fibras")
    plt.ylabel("Inclusión")
    plt.title("Métrica de Inclusión")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/inclusion.png", dpi=300)


def main():
    evaluate_octree_performance()

    # Configurar entorno
    N = [50, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    bundle, vertex_lh, polygons_lh, bounds, octree = setup_environment(N)

    # Evaluar algoritmos
    evaluate_algorithms(N, bundle, vertex_lh, polygons_lh, bounds, octree)

if __name__ == '__main__':
    main()