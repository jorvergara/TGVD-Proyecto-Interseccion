# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:55:16 2024

@author: User
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import find
from scipy.sparse.linalg import spsolve

def cotangent_laplacian(vertices, faces):
    """Calcula los pesos cotangentes para la laplaciana."""
    
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0 = vertices[f0]
    v1 = vertices[f1]
    v2 = vertices[f2]
    
    l0 = np.linalg.norm(v1 - v2, axis=1)
    l1 = np.linalg.norm(v2 - v0, axis=1)
    l2 = np.linalg.norm(v0 - v1, axis=1)
    
    s = (l0 + l1 + l2)*0.5
    area = np.sqrt(s*(s-l0)*(s-l1)*(s-l2))
 
    cot01 = (l0**2 + l1**2 - l2**2)/area/2
    cot12 = (l1**2 + l2**2 - l0**2)/area/2 
    cot20 = (l2**2 + l0**2 - l1**2)/area/2
    
    diag0 = -cot01-cot20
    diag1 = -cot01-cot12
    diag2 = -cot20-cot12;
    
    II =  np.concatenate([f0, f1, f1, f2, f2, f0, f0, f1, f2])
    JJ = np.concatenate([f1, f0, f2, f1, f0, f2, f0, f1, f2])
    V = np.concatenate([cot01, cot01, cot12, cot12, cot20, cot20, diag0, diag1, diag2])
    
    n = len(vertices)
    L = csr_matrix((V, (II, JJ)), shape=(n, n))
    return L


def spherical_conformal_map(vertices, faces):
    """Calcula el mapeo conforme esférico."""
    check_genus_zero(vertices, faces)   
    bigtri = find_most_regular_triangle(vertices, faces)
    
    nv = vertices.shape[0]  # Número de vértices

    # Calcular la matriz Laplaciana cotangente
    M = cotangent_laplacian(vertices, faces)  # Esta función debe estar definida

    # Obtener los índices de los vértices del triángulo grande
    p1, p2, p3 = faces[bigtri]
    
    # Fijar los vértices correspondientes
    fixed = np.array([p1, p2, p3])

    # Encuentra los elementos no ceros relacionados con los vértices fijos
    
    mrow, mcol, mval = find(M[fixed, :])

    # Ajustar la matriz Laplaciana
    M += M - csr_matrix((mval, (fixed[mrow], mcol)), shape=(nv, nv))
    M += csr_matrix((np.ones(3), (fixed, fixed)), shape=(nv, nv))
    
    ### Set the boundary condition for big triangle
    # Coordenadas arbitrarias de dos puntos del triángulo
    x1, y1 = 0, 0
    x2, y2 = 1, 0

    # Calcular los vectores a y b
    a = vertices[p2] - vertices[p1]
    b = vertices[p3] - vertices[p1]
    
    # Calcular el seno del ángulo entre los vectores a y b
    sin1 = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Altura original del triángulo
    ori_h = np.linalg.norm(b) * sin1

    # Razón de escalamiento
    ratio = np.linalg.norm([x1 - x2, y1 - y2]) / np.linalg.norm(a)

    # Coordenadas del tercer vértice
    y3 = ori_h * ratio
    x3 = np.sqrt(np.linalg.norm(b)**2 * ratio**2 - y3**2)
    
    # Crear los vectores c y d con las condiciones de frontera
    c = np.zeros(nv)
    c[p1], c[p2], c[p3] = x1, x2, x3

    d = np.zeros(nv)
    d[p1], d[p2], d[p3] = y1, y2, y3
    
    z = spsolve(M, c + 1j * d)
    z = z-np.mean(z)
    
    # Proyección estereográfica inversa
    abs_z_squared = np.abs(z)**2
    S = np.column_stack([
        2 * np.real(z) / (1 + abs_z_squared),
        2 * np.imag(z) / (1 + abs_z_squared),
        (-1 + abs_z_squared) / (1 + abs_z_squared)
    ])
    
    # Calcular las coordenadas complejas para cada vértice
    w = S[:, 0] / (1 + S[:, 2]) + 1j * S[:, 1] / (1 + S[:, 2])
    
    
    # Ordenar triángulos según la suma de valores absolutos de z para cada vértice
    triangle_scores = np.abs(z[faces[:, 0]]) + np.abs(z[faces[:, 1]]) + np.abs(z[faces[:, 2]])
    index = np.argsort(triangle_scores)
    # Seleccionar el triángulo más al sur que no sea el triángulo grande
    inner = index[0]
    if inner == bigtri:
        inner = index[1]
        
        
    # Tamaño del triángulo norte (bigtri)
    NorthTriSide = (
        (np.abs(z[faces[bigtri, 0]] - z[faces[bigtri, 1]]) +
         np.abs(z[faces[bigtri, 1]] - z[faces[bigtri, 2]]) +
         np.abs(z[faces[bigtri, 2]] - z[faces[bigtri, 0]])) / 3
    )

    # Tamaño del triángulo sur (inner)
    SouthTriSide = (
        (np.abs(w[faces[inner, 0]] - w[faces[inner, 1]]) +
         np.abs(w[faces[inner, 1]] - w[faces[inner, 2]]) +
         np.abs(w[faces[inner, 2]] - w[faces[inner, 0]])) / 3
    )
    
    # Escalar z para obtener la mejor distribución
    z = z * (np.sqrt(NorthTriSide * SouthTriSide)) / NorthTriSide
    
    # Proyección estereográfica inversa
    abs_z_squared = np.abs(z)**2
    S = np.column_stack([
        2 * np.real(z) / (1 + abs_z_squared),
        2 * np.imag(z) / (1 + abs_z_squared),
        (-1 + abs_z_squared) / (1 + abs_z_squared)
    ])
    
    # if sum(sum(isnan(S))) ~= 0
    # % if harmonic map fails due to very bad triangulations, use tutte map
    # S = spherical_tutte_map(f,bigtri);
    # end
    
    # Ordenar puntos por la coordenada z (tercera columna de S)
    I = np.argsort(S[:, 2])

    # Número de puntos cercanos al polo sur que serán fijados
    fixnum = max(round_half_up(len(vertices) / 10), 3)
    fixed = I[:min(len(vertices), fixnum)]

    # Proyección estereográfica desde el polo sur
    P = np.column_stack([S[:, 0] / (1 + S[:, 2]), S[:, 1] / (1 + S[:, 2])])
    
    #### REVISAR BELTRAMI COEF
    mu = beltrami_coefficient(P, faces, vertices)
    map_coords = linear_beltrami_solver(P, faces, mu, fixed, P[fixed, :])
    
    # print(np.isnan(map_coords).sum())
    
    z = map_coords[:, 0] + 1j * map_coords[:, 1]

    # Proyección estereográfica inversa desde el polo sur
    map_3d = np.column_stack([
        2 * np.real(z) / (1 + np.abs(z)**2),
        2 * np.imag(z) / (1 + np.abs(z)**2),
        -(np.abs(z)**2 - 1) / (1 + np.abs(z)**2)
    ])
    return map_3d

def check_genus_zero(vertices, faces):
    if len(vertices) - 3 * len(faces) / 2 + len(faces) != 2:
        raise ValueError("The mesh is not a genus-0 closed surface.")
        
def find_most_regular_triangle(vertices, faces):
    # Reorganizar los vértices según las caras
    temp = vertices[faces.flatten()]
    
    # Calcular las longitudes de los lados de los triángulos
    e1 = np.sqrt(np.sum((temp[1::3] - temp[2::3]) ** 2, axis=1))
    e2 = np.sqrt(np.sum((temp[0::3] - temp[2::3]) ** 2, axis=1))
    e3 = np.sqrt(np.sum((temp[0::3] - temp[1::3]) ** 2, axis=1))
    
    # Calcular la regularidad de los triángulos
    total_edge_length = e1 + e2 + e3
    regularity = (
        np.abs(e1 / total_edge_length - 1 / 3) +
        np.abs(e2 / total_edge_length - 1 / 3) +
        np.abs(e3 / total_edge_length - 1 / 3)
    )
    
    # Encontrar el triángulo más regular
    bigtri = np.argmin(regularity)
    return bigtri

def round_half_up(x):
    if x >=0.5:
        return np.ceil(x).astype('int')
    return np.floor(x).astype('int')


def beltrami_coefficient(vertices, faces, mapping):
    """
    Calcula el coeficiente de Beltrami de un mapeo.
    
    Parameters:
        vertices (numpy array): Coordenadas de los vértices (n, 3).
        faces (numpy array): Índices de los vértices que forman triángulos (m, 3).
        mapping (numpy array): Coordenadas mapeadas (n, 3).
    
    Returns:
        numpy array: Coeficiente de Beltrami para cada triángulo.
    """
    num_faces = len(faces)
    
    # Índices para construir matrices dispersas
    Mi = np.repeat(np.arange(num_faces), 3)
    Mj = faces.flatten()

    # Lados de los triángulos en 2D
    e1 = vertices[faces[:, 2], :2] - vertices[faces[:, 1], :2]
    e2 = vertices[faces[:, 0], :2] - vertices[faces[:, 2], :2]
    e3 = vertices[faces[:, 1], :2] - vertices[faces[:, 0], :2]

    # Áreas de los triángulos
    area = (-e2[:, 0] * e1[:, 1] + e1[:, 0] * e2[:, 1])/ 2
    area = np.stack([area, area, area])

    # Construir matrices dispersas para derivadas
    Mx = (np.stack([e1[:, 1], e2[:, 1], e3[:, 1]])/area/2).T.flatten()
    My = -(np.stack([e1[:, 0], e2[:, 0], e3[:, 0]])/area/2).T.flatten()


    Dx = csr_matrix((Mx, (Mi, Mj)), shape=(num_faces, len(vertices)))
    Dy = csr_matrix((My, (Mi, Mj)), shape=(num_faces, len(vertices)))

    dXdu = Dx * mapping[:, 0]
    dXdv = Dy * mapping[:, 0]
    dYdu = Dx * mapping[:, 1]
    dYdv = Dy * mapping[:, 1]
    dZdu = Dx * mapping[:, 2]
    dZdv = Dy * mapping[:, 2]

    # Coeficientes métricos
    E = dXdu**2 + dYdu**2 + dZdu**2
    G = dXdv**2 + dYdv**2 + dZdv**2
    F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv

    # Cálculo del coeficiente de Beltrami
    mu = (E - G + 2j * F) / (E + G + 2 * np.sqrt(E * G - F**2))
    
    return mu

def linear_beltrami_solver(vertices, faces, mu, landmark, target):
    """
    Soluciona el problema lineal de Beltrami.
    
    Parameters:
        vertices (numpy array): Coordenadas de los vértices (n, 3).
        faces (numpy array): Índices de los vértices que forman triángulos (m, 3).
        mu (numpy array): Coeficiente de Beltrami (m,).
        landmark (numpy array): Índices de los vértices de referencia.
        target (numpy array): Coordenadas objetivo para los landmarks (k, 2).
    
    Returns:
        numpy array: Coordenadas del mapeo (n, 2).
    """
    # Calcular los coeficientes a, b, g
    af = (1 - 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    bf = -2 * np.imag(mu) / (1 - np.abs(mu)**2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)

    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Calcular componentes ux y uy
    uxv0 = vertices[f1, 1] - vertices[f2, 1]
    uyv0 = vertices[f2, 0] - vertices[f1, 0]
    uxv1 = vertices[f2, 1] - vertices[f0, 1]
    uyv1 = vertices[f0, 0] - vertices[f2, 0]
    uxv2 = vertices[f0, 1] - vertices[f1, 1]
    uyv2 = vertices[f1, 0] - vertices[f0, 0]

    # Calcular las longitudes de los lados y el área
    l = np.stack([np.sqrt(uxv0**2 + uyv0**2), np.sqrt(uxv1**2 + uyv1**2), np.sqrt(uxv2**2 + uyv2**2)],axis =1)
    s = np.sum(l, axis = 1) * 0.5
    area = np.sqrt(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]))

    # Construir los valores para la matriz dispersa
    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area
    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area

    I = np.hstack([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    J = np.hstack([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    V = np.hstack([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / 2

    # Construir matriz dispersa A
    A = csr_matrix((V, (I, J)), shape=(len(vertices), len(vertices)))

    # Calcular el término derecho b
    target_complex = target[:, 0] + 1j * target[:, 1]
    b = -A[:, landmark] * target_complex
    b[landmark] = target_complex

    # Ajustar filas y columnas correspondientes a los landmarks
    A[landmark, :] = 0
    A[:, landmark] = 0
    A += csr_matrix((np.ones(len(landmark)), (landmark, landmark)), shape=A.shape)

    # Resolver el sistema lineal
    map_complex = spsolve(A, b)
    map_coords = np.column_stack([np.real(map_complex), np.imag(map_complex)])

    return map_coords


def main():
    vertices = np.array([[131.51915 , 118.89038 , 131.17783 ],
         [113.556595, 150.23898 , 104.5649  ],
         [121.34059 , 185.22395 , 111.4095  ],
         [115.212685,  55.797153,  59.81363 ],
         [113.51371 ,  54.248756,  58.19632 ],
         [114.6446  ,  54.879593,  60.346653]])
    faces = np.array([[ 0, 1, 2],[1, 2, 3],[2, 3, 4],[3, 4, 5]])

    mapping = np.array([[ 12.66549439,   5.77015227,  41.04887874],
                         [ -5.29706054,  37.11875457,  14.43595607],
                         [  2.48693604,  72.10372466,  21.28055293],
                         [ -3.64097076, -57.32307511, -30.31531804],
                         [ -5.33994536, -58.87147218, -31.93262761],
                         [ -4.20905547, -58.24063569, -29.78229421]])

    num_faces = len(faces)

    # Índices para construir matrices dispersas
    Mi = np.repeat(np.arange(num_faces), 3)
    Mj = faces.flatten()

    # Lados de los triángulos en 2D
    e1 = vertices[faces[:, 2], :2] - vertices[faces[:, 1], :2]
    e2 = vertices[faces[:, 0], :2] - vertices[faces[:, 2], :2]
    e3 = vertices[faces[:, 1], :2] - vertices[faces[:, 0], :2]

    # Áreas de los triángulos
    area = (-e2[:, 0] * e1[:, 1] + e1[:, 0] * e2[:, 1])/ 2
    area = np.stack([area, area, area])

    # Construir matrices dispersas para derivadas
    Mx = (np.stack([e1[:, 1], e2[:, 1], e3[:, 1]])/area/2).T.flatten()
    My = -(np.stack([e1[:, 0], e2[:, 0], e3[:, 0]])/area/2).T.flatten()


    Dx = csr_matrix((Mx, (Mi, Mj)), shape=(num_faces, len(vertices)))
    Dy = csr_matrix((My, (Mi, Mj)), shape=(num_faces, len(vertices)))

    dXdu = Dx * mapping[:, 0]
    dXdv = Dy * mapping[:, 0]
    dYdu = Dx * mapping[:, 1]
    dYdv = Dy * mapping[:, 1]
    dZdu = Dx * mapping[:, 2]
    dZdv = Dy * mapping[:, 2]

    # Coeficientes métricos
    E = dXdu**2 + dYdu**2 + dZdu**2
    G = dXdv**2 + dYdv**2 + dZdv**2
    F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv

    # Cálculo del coeficiente de Beltrami
    mu = (E - G + 2j * F) / (E + G + 2 * np.sqrt(E * G - F**2))

    # Calcular los coeficientes a, b, g
    af = (1 - 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    bf = -2 * np.imag(mu) / (1 - np.abs(mu)**2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)

    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Calcular componentes ux y uy
    uxv0 = vertices[f1, 1] - vertices[f2, 1]
    uyv0 = vertices[f2, 0] - vertices[f1, 0]
    uxv1 = vertices[f2, 1] - vertices[f0, 1]
    uyv1 = vertices[f0, 0] - vertices[f2, 0]
    uxv2 = vertices[f0, 1] - vertices[f1, 1]
    uyv2 = vertices[f1, 0] - vertices[f0, 0]



    # Calcular las longitudes de los lados y el área
    l = np.stack([np.sqrt(uxv0**2 + uyv0**2), np.sqrt(uxv1**2 + uyv1**2), np.sqrt(uxv2**2 + uyv2**2)],axis =1)
    s = np.sum(l, axis = 1) * 0.5
    area = np.sqrt(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]))

    # Construir los valores para la matriz dispersa
    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area
    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area

    I = np.hstack([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    J = np.hstack([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    V = np.hstack([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / 2

    # Construir matriz dispersa A
    A = csr_matrix((-V, (I, J)), shape=(len(vertices), len(vertices)))