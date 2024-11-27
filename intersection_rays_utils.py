# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:51:58 2024

@author: Jorge
"""
import numpy as np
def ray_triangle_intersection_matrix(ray_start, ray_vec, triangle):
    
    eps = 1e-8
    v1,v2,v3 = np.split(triangle, 3, axis = 1)
    
    edge1 = v2 - v1
    edge2 = v3 - v1
    ray_vec_normalized = ray_vec / np.linalg.norm(ray_vec, axis=1, keepdims=True)
    
    pvec = np.cross(ray_vec_normalized, edge2)
    
    det = np.sum(pvec*edge1,axis = 2)
    mask = np.abs(det) >= eps
    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]
    
    tvec = ray_start - v1
    
    u = np.sum(tvec*pvec, axis = 2) * inv_det
    
    qvec = np.cross(tvec, edge1)
    
    v = np.sum(ray_vec_normalized*qvec, axis = 2) * inv_det
    
    t = np.sum(qvec*edge2,axis =2) * inv_det
    

    ray_length = np.linalg.norm(ray_vec, axis=1)
    ray_length = ray_length[None, :]
    

    result = np.bitwise_not(np.logical_or.reduce([u < 0.0, u > 1.0, v < 0.0, u + v > 1.0, t<eps, t > ray_length]))
    return result

def ray_triangle_intersection_vector(ray_start, ray_vec, triangle):
    
    eps = 1e-8
    v1,v2,v3 = np.split(triangle, 3, axis = 1)

    
    edge1 = v2 - v1
    edge2 = v3 - v1
    
    ray_vec_normalized = ray_vec / np.linalg.norm(ray_vec, axis=0, keepdims=True)
    
    pvec = np.cross(ray_vec_normalized, edge2)
    
    det = np.sum(pvec*edge1,axis = 2)
    mask = np.abs(det) >= eps
    inv_det = np.zeros_like(det)
    inv_det[mask] = 1.0 / det[mask]
    
    
    tvec = ray_start - v1
    
    u = np.sum(tvec*pvec, axis = 2) * inv_det
    
    qvec = np.cross(tvec, edge1)
    
    v = np.sum(ray_vec_normalized*qvec, axis = 2) * inv_det
    
    t = np.sum(qvec*edge2,axis =2) * inv_det
    
    ray_length = np.linalg.norm(ray_vec, axis=0, keepdims=True)
    
   
    result = np.bitwise_not(np.logical_or.reduce([u < 0.0, u > 1.0, v < 0.0, u + v > 1.0, t<eps, t > ray_length]))
    return result