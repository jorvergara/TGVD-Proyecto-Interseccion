#Joaquín 2023 ;) 

import os
import subprocess as sp
from time import time
import numpy as np
import bundleTools as BT
import bundleTools3 as BT3
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import nilearn
from nilearn import surface
#import HierClustTools as hct
from collections import defaultdict
import pickle


#Ejecutar intersección. 
#arg1: Directorio con mallados lh y rh .obj
#arg2: Directorio con archivos .bundles a intersectar.
#arg3: Directorio de resultados
#Return: Escribe datos de intersección en archivo binario .intersectiondata en el siguiente formato: InTri, FnTri, InPoints, FnPoints, fib_index, fiber_class
def intersection(meshes_path, bundles_path, results_path):
    # print("Intersection...")

    t=time()

    #---------- Setup (mallado y creación de carpetas) ----------#
    Lhemi_path = meshes_path+ '/lh.obj'
    Rhemi_path = meshes_path+ '/rh.obj'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #sp.call(['intersection/make']);  #Compilación de códigos de intersección (tools.h, main.cpp, etc.)
    # print(['intersection/./interx', Lhemi_path, Rhemi_path, bundles_path, results_path])
    sp.call(['intersection/./interx', Lhemi_path, Rhemi_path, bundles_path, results_path])

    # print('Tiempo: ' + str(time()-t)+'[s]' )

#Leer datos de intersección.
#arg1: Archivo de intersección .intersectiondata
#Return: Triángulos iniciales, triángulos finales, puntos iniciales, puntos finales, indice de la fibra que intersecta esos triángulos, mallados intersectados (1: LH-LH, 2: RH-RH, 3: LH-RH, 4: RH-LH)
def read_intersection(infile):
    #print("Read intersection...")

    f = open(infile, 'rb');

    total_triangles = np.frombuffer( f.read( 4 ), np.uint32 )[ 0 ];

    InTri = np.frombuffer( f.read( 4 * total_triangles), np.uint32 );
    FnTri = np.frombuffer( f.read( 4 * total_triangles), np.uint32 );

    InPoints = np.frombuffer( f.read( 4 * 3 * total_triangles), np.float32 ).reshape(-1,3);
    FnPoints = np.frombuffer( f.read( 4 * 3 * total_triangles), np.float32 ).reshape(-1,3);

    fib_index = np.frombuffer( f.read( 4 * total_triangles), np.uint32 );
    fiber_class = np.frombuffer( f.read( 4 * total_triangles), np.uint32);

    f.close();
    return InTri, FnTri, InPoints, FnPoints, fib_index, fiber_class;

#Seleccionar los índices de cada clase 
#arg1: fiber_class obtenido de read_intersection()
#Return: Listas que contiene los índices de cada clase.
def return_fiberclass(fib_class):
    #print("Fiber class selection...")
    fibs_lh = list(np.where(np.array(fib_class) == 1)[0])
    fibs_rh = list(np.where(np.array(fib_class) == 2)[0])
    fibs_inter_l_r = list(np.where(np.array(fib_class) == 3)[0])
    fibs_inter_r_l = list(np.where(np.array(fib_class) == 4)[0]) 
    
    return fibs_lh, fibs_rh, fibs_inter_l_r, fibs_inter_r_l


#Escribir alguna clase en un archivo .bundles
#arg1: Una lase de fibras obtenidas de return_fiberclass()
#arg2: Archivo de fibras .bundles
#arg3: fib_index obtenido de read_intersection()
#arg4: Directorio de resultados
#arg5: Nombre del archivo a escribir.
#Return: Escribe la clase dada por arg1 en un archivo .bundles
def write_fiberclass_files(fiber_class, all_fibers_path, fib_index, path_result, name):
    print("Writing fiber class files")
    
    all_fibers = BT.read_bundle(str(all_fibers_path))
    fibers = []
    for n in fiber_class:
        fibers.append(all_fibers[fib_index[n]])

    if not os.path.exists(path_result):
            os.makedirs(path_result)
        
    BT3.write_bundle(str(path_result) + str(name) + ".bundles", fibers, 'points')

#Connectome with gifti parcels.
#arg1: Archivo de intersección .intersectiondata
#arg2: Directorio con mallados lh y rh .obj
#arg3: Archivo .gii de parcelas lh (se lee como una lista donde cada índice es un vértices de la superficie cortical y los valores son la etiqueta de la parcela (int))
#arg4: Idem pero de RH
#arg5: Directorio de resultados.
#arg6: Nombre de archivo para guardar imágen (png) del conectoma y datos (txt)
#Return: Escribe datos png, txt de la matriz y entrega un diccionario con los perfiles de conectividad con el siguiente formato: {'ROI1_ROI2': [fib_indexes_connecting_RO1_ROI2], ...}
#NOTA: ESTA FUNCIÓN ESTÁ ADAPTADA PARA TRABAJAR CON LOS ARCHIVOS DE DESIKAN-KILLIANY DE NARCISO. Para trabajar con otros archivos .gii revisar la función
def connectome_gifti(intersection_file, meshes_path, parcels_lh, parcels_rh, results_path, name):
    print("Computing structural connectome...")

    InTri, FnTri, InPoints, FnPoints, fib_index, fiber_class = read_intersection(intersection_file)
    fibs_lh, fibs_rh, fibs_inter_l_r, fibs_inter_r_l = return_fiberclass(fiber_class)
    #print(intersection_file)
    #print(InTri)

    if not os.path.exists(results_path):
            os.makedirs(results_path)

    #Load gifti data.
    labels_arr_lh = nilearn.surface.load_surf_data(parcels_lh)
    labels_arr_rh = nilearn.surface.load_surf_data(parcels_rh)

    #Load meshes
    Lhemi_path = meshes_path + "lh.obj"
    Lvertex, Lpolygons = BT.read_mesh_obj(Lhemi_path)

    Rhemi_path = meshes_path + "rh.obj"
    Rvertex, Rpolygons = BT.read_mesh_obj(Rhemi_path)

    #Set the label for each triangle
    TriLabel_lh = []
    for polygon in Lpolygons:
        
        #values of vertices
        vert1 = polygon[0]
        vert2 = polygon[1]
        vert3 = polygon[2]

        #labels of vertices
        label_vert1 = labels_arr_lh[vert1]
        label_vert2 = labels_arr_lh[vert2]
        label_vert3 = labels_arr_lh[vert3]

        #We need at least 2 vertices with the same label to label the whole triangle.
        #If not, the triangle is labeled with -1 (the label -1 will be discarded)
        if label_vert1 == label_vert2 or label_vert1 == label_vert3:
            TriLabel_lh.append(label_vert1)
            
        elif label_vert2 == label_vert3:
            TriLabel_lh.append(label_vert2)

        else:
            TriLabel_lh.append(-1)
            
    #Set the label for each triangle
    TriLabel_rh = []
    for polygon in Rpolygons:
        
        #values of vertices
        vert1 = polygon[0]
        vert2 = polygon[1]
        vert3 = polygon[2]

        #labels of vertices
        label_vert1 = labels_arr_rh[vert1]
        label_vert2 = labels_arr_rh[vert2]
        label_vert3 = labels_arr_rh[vert3]

        #We need at least 2 vertices with the same label to label the whole triangle.
        #If not, the triangle is labeled with -1 (the label -1 will be discarded)
        if label_vert1 == label_vert2 or label_vert1 == label_vert3:
            TriLabel_rh.append(label_vert1)
            
        elif label_vert2 == label_vert3:
            TriLabel_rh.append(label_vert2)

        else:
            TriLabel_rh.append(-1)
    
    len_lh = len(list(set(labels_arr_lh)))
    len_rh = len(list(set(labels_arr_rh)))

    #Connectivity matrix calculation 
    Conmatrix = np.zeros(((len_lh + len_rh),(len_lh + len_rh))) # The labels starts at 1, not 0 
    condict = defaultdict(list)

    for i in range(0, len(fib_index)):   

        f_index = fib_index[i]
        class_fib = fiber_class[i]
        intri_actual = InTri[i]
        fntri_actual = FnTri[i]
        
        if class_fib == 1:
            label1 = TriLabel_lh[intri_actual]
            label2 = TriLabel_lh[fntri_actual]
            
            if label1 == -1 or label2 == -1:
                continue

            condict['L' + str(label1) + '_' + 'L' + str(label2)].append(f_index)
            Conmatrix[label1-1][label2-1] +=1
            Conmatrix[label2-1][label1-1] +=1

        elif class_fib == 2:
            label1 = TriLabel_rh[intri_actual]
            label2 = TriLabel_rh[fntri_actual]
            
            if label1 == -1 or label2 == -1:
                continue
            
            condict['R' + str(label1) + '_' + 'R' + str(label2)].append(f_index)
            Conmatrix[(label1-1)+len_lh][(label2-1)+len_lh] +=1
            Conmatrix[(label2-1)+len_lh][(label1-1)+len_lh] +=1
            
        elif class_fib == 3:
            label1 = TriLabel_lh[intri_actual]
            label2 = TriLabel_rh[fntri_actual]
            
            if label1 == -1 or label2 == -1:
                continue
            
            condict['L' + str(label1) + '_' + 'R' + str(label2)].append(f_index)
            Conmatrix[label1-1][(label2-1)+len_lh] +=1
            Conmatrix[(label2-1) + len_lh][label1-1] +=1

        elif class_fib == 4:
            label1 = TriLabel_rh[intri_actual]
            label2 = TriLabel_lh[fntri_actual]
            
            if label1 == -1 or label2 == -1:
                continue
            
            condict['R' + str(label1) + '_' + 'L' + str(label2)].append(f_index)
            Conmatrix[(label1-1)+len_lh][(label2-1)] +=1
            Conmatrix[(label2-1)][(label1-1)+len_lh] +=1

    np.savetxt(results_path + name +".txt", Conmatrix)

    with open(results_path + name + ".pkl", 'wb') as f:
        pickle.dump(condict, f)

    plt.figure()
    plt.title(name)
    fig_ConMatrix = sns.heatmap(Conmatrix, cmap = 'gnuplot')   
    plt.savefig(results_path + name + '.png')


#load parcelsdata parcels
def load_parcels (parcels_path): 

    parcels = dict()
        
    for Dir in os.listdir(parcels_path):
        parcels[Dir.split('.')[0]] = BT.read_parcels(parcels_path + "/" + Dir)
            
    return parcels


#Connectome with parcelsdata parcels
def connectome_parcelsdata(intersection_file, meshes_path, parcels_lh, parcels_rh, results_path, name):

    #Cargar parcelación y ver cuantas parcelas tiene.
    parcels_dict_lh = load_parcels(parcels_lh)
    len_lh = len(parcels_dict_lh)

    parcels_dict_rh = load_parcels(parcels_rh)
    len_rh = len(parcels_dict_rh)

    original_parcels_lh = parcels_dict_lh
    original_parcels_rh = parcels_dict_rh

    InTri, FnTri, InPoints, FnPoints, fib_index, fiber_class = read_intersection(intersection_file)
    fibs_lh, fibs_rh, fibs_inter_l_r, fibs_inter_r_l = return_fiberclass(fiber_class)

    if not os.path.exists(results_path):
            os.makedirs(results_path)

    #Load meshes
    Lhemi_path = meshes_path + "lh.obj"
    Lvertex, Lpolygons = BT.read_mesh_obj(Lhemi_path)

    Rhemi_path = meshes_path + "rh.obj"
    Rvertex, Rpolygons = BT.read_mesh_obj(Rhemi_path)
    
    #Crear nuevos labels desde 0 a n... para la matriz de conectividad
    corresponding_labels_lh = (parcels_dict_lh.keys())
    new_labels_lh = [str(i2) for i2 in range(len_lh)]
    
    corresponding_labels_rh = (parcels_dict_rh.keys())
    new_labels_rh = [str(i2) for i2 in range(len_rh)]
    
    #changing keys of dictionary
    parcels_dict_lh = dict(zip(new_labels_lh, list(parcels_dict_lh.values())))
    parcels_dict_rh = dict(zip(new_labels_rh, list(parcels_dict_rh.values())))

    #TriLabel tiene las etiquetas de cada parcela (puede ser más de una).
    TriLabel_lh = []
    TriLabel_rh = []
    for i in range(len(Lpolygons)):
        TriLabel_lh.append([])
        TriLabel_rh.append([])

    for k,v in parcels_dict_lh.items():
        for tri in v:
            TriLabel_lh[tri].append(k)
            
    for k,v in parcels_dict_rh.items():
        for tri in v:
            TriLabel_rh[tri].append(k)
    
    matriz_size = len_lh + len_rh
    Conmatrix = np.zeros((matriz_size,matriz_size))
    condict = defaultdict(list)

    for i in range(0, len(fib_index)):
        f_index = fib_index[i]
        class_fib = fiber_class[i]
        intri_actual = InTri[i]
        fntri_actual = FnTri[i]
        
        #lh  lh
        if class_fib == 1:
            label1 = [int(l1) for l1 in TriLabel_lh[intri_actual]]
            label2 = [int(l2) for l2 in TriLabel_lh[fntri_actual]]
            
            if (label1 and label2) == False:
                continue

            else:
                for x in range(len(label1)):
                    for y in range(len(label2)):

                        #list(original_parcels_lh.keys())[int(label1[x])] + '_' + list(original_parcels_lh.keys())[int(label2[x])]
                        
                        condict['L' + str(list(original_parcels_lh.keys())[int(label1[x])]) + '_' + 'L' + str(list(original_parcels_lh.keys())[int(label2[y])])].append(f_index)

                        if label1[x] != label2[y]:

                            Conmatrix[label1[x]][label2[y]] +=1
                            Conmatrix[label2[y]][label1[x]] +=1
                            
                        elif label1[x] == label2[y]:
                            Conmatrix[label1[x]][label2[y]] +=1
        
        #rh rh
        elif class_fib == 2:
            label1 = [int(l1) for l1 in TriLabel_rh[intri_actual]]
            label2 = [int(l2) for l2 in TriLabel_rh[fntri_actual]]
            
            if (label1 and label2) == False:
                continue
                
            else:
                for x in range(len(label1)):
                    for y in range(len(label2)):
                        
                        condict['R' + str(list(original_parcels_rh.keys())[int(label1[x])]) + '_' + 'R' + str(list(original_parcels_rh.keys())[int(label2[y])])].append(f_index)

                        if label1[x] != label2[y]:
                            Conmatrix[label1[x]+len_lh][label2[y]+len_lh] +=1
                            Conmatrix[label2[y]+len_lh][label1[x]+len_lh] +=1
                            
                        elif label1[x] == label2[y]:
                            Conmatrix[label1[x]+len_lh][label2[y]+len_lh] +=1

            
        elif class_fib == 3:
            label1 = [int(l1) for l1 in TriLabel_lh[intri_actual]]
            label2 = [int(l2) for l2 in TriLabel_rh[fntri_actual]]
            
            if (label1 and label2) == False:
                continue
                
            else:
                for x in range(len(label1)):
                    for y in range(len(label2)):

                        condict['L' + str(list(original_parcels_lh.keys())[int(label1[x])]) + '_' + 'R' + str(list(original_parcels_rh.keys())[int(label2[y])])].append(f_index)

                        Conmatrix[label1[x]][label2[y]+len_lh] +=1
                        Conmatrix[label2[y]+len_lh][label1[x]] +=1
            
        elif class_fib == 4:
            label1 = [int(l1) for l1 in TriLabel_rh[intri_actual]]
            label2 = [int(l2) for l2 in TriLabel_lh[fntri_actual]]
            
            if (label1 and label2) == False:
                continue
                
            else:
                for x in range(len(label1)):
                    for y in range(len(label2)):

                        condict['R' + str(list(original_parcels_rh.keys())[int(label1[x])]) + '_' + 'L' + str(list(original_parcels_lh.keys())[int(label2[y])])].append(f_index)

                        Conmatrix[label1[x]+len_lh][label2[y]] +=1
                        Conmatrix[label2[y]][label1[x]+len_lh] +=1        
    
    np.savetxt(results_path + name +".txt", Conmatrix)

    with open(results_path + name + ".pkl", 'wb') as f:
        pickle.dump(condict, f)

    plt.figure()
    plt.title(name)
    fig_ConMatrix = sns.heatmap(Conmatrix, cmap = 'gnuplot')   
    plt.savefig(results_path + name + '.png')


#Escribe las fibras que conectan dos ROIs
#arg1: connectivity_profile_dict obtenido de connectome_gifti() o connectome_parcelsdata()
#arg2: Región de interés 1
#arg3: Región de interés 2
#arg4: Archivo del archivo bundles intersectado.
#arg5: Nombre y path para guardar las fibras
def write_2roi_bundles(connectivity_profile_dict, roi1, roi2, all_fibers_path, result_path):
    key = str(roi1) + "_" + str(roi2)
    indexes = connectivity_profile_dict[key]

    fibers = BT.read_bundle(str(all_fibers_path))
    
    bundle = []
    for index in indexes:
        bundle.append(fibers[index])

    BT3.write_bundle(str(result_path) + '.bundles', bundle, 'points')
