# Índice
Los códigos presentados a continuación constituyen las 3 partes del algoritmo de intersección cluster-corteza considerando conexiones ínter-hemisféricas. Por lo tanto, debiesen ejecutarse en forma secuencial, donde la salida de uno es la entrada del siguiente; finalmente, la salida del último código es el conjunto de todas las intersecciones encontradas.

1. [Fiber-Mesh Intersection (Modified)](#intersection)
2. [Bundle Classification](#bundle_classification)
3. [Intersection Classification](#intersection_classification)

## Fiber-Mesh Intersection (Modified) <a name="intersection"></a>
Este algoritmo es una modificación de aquel utilizado en [1]. Específicamente, esta versión evalúa la intersección con un hemisferio a la vez, en lugar de ambos al mismo tiempo; además, esta versión también considera el cálculo de los perfiles de conectividad, por lo que no descarta las fibras que intersectan con la corteza en sólo uno de sus extremos (a diferencia del original, que sólo considera aquellas que conectan en ambas extremidades). Para más detalles, leer el extracto de MT incluido en la carpeta.

### Ejemplo de uso
Ejecutar (desde la terminal o desde el código fuente, por ejemplo en spyder):
```
python3 interx.py
```
A diferencia de los algoritmos de preprocesamiento, este código no cuenta con parámetros modificables desde terminal. Por lo tanto, las rutas de los mallados corticales y de los clusters deben ser modificadas directamente en el archivo interx.py.

Por defecto, el algoritmo se ejecuta 79 veces, correspondiente a los 79 sujetos de la base de datos ARCHI. Esto también se puede modificar en el código interx.py

### Archivos de entrada
- Mallados corticales en formato .obj, un archivo por cada hemisferio.
- Fascículos/clusters en formato .bundles y .bundlesdata, resampleados a 21 puntos por fibra, y separados en un archivo por cada fascículo/cluster.

### Archivos de salida
- Carpeta "/intersection/", que contiene los datos de intersección de cada fascículo en cada caso (L_hemi_direct, R_hemi_direct, L_hemi_inverse, R_hemi_inverse), almacenados en archivos binarios (.intersectiondata), los cuales son:

  - Índice del triángulo que intersecta con el extremo inicial de la fibra.
  - Índice del triángulo que intersecta con el extremo final de la fibra.*
  - Punto exacto de intersección con el extremo inicial de la fibra.
  - Punto exacto de intersección con el extremo final de la fibra.
  - Índice de la fibra que intersecta con los triángulos.

\* Nota: Debido a la naturaleza del algoritmo, las fibras que se consideran potencialmente ínter-hemisferio sólo intersectan en su extremo inicial; es decir, no intersectan con ningún triángulo en su extremo final. En estos casos se guardó un índice de 1000000 para dicho triángulo, y un punto exacto de intersección de (0,0,0). Esto será útil en los dos códigos posteriores.

- Carpeta "/membership/", que contiene los datos de conectividad/pertenencia de cada fascículo en cada caso (L_hemi_direct, R_hemi_direct, L_hemi_inverse, R_hemi_inverse), almacenados en archivos binarios (.membershipdata). Los valores posibles son:

  - (1,1): Cluster intersecta en ambos extremos.
  - (1,0): Cluster sólo intersecta en su extremo inicial.
  - (0,0): Cluster no intersecta.

## Bundle Classification <a name="bundle_classification"></a>
Este algoritmo intenta clasificar los clusters en cuatro categorías distintas:

  - L-L: Clusters intra-hemisferio izquierdo, es decir, sus fibras intersectan el hemisferio izquierdo en ambos extremos.
  - L-R: Clusters ínter-hemisferio izquierda-derecha, es decir, sus fibras intersectan al hemisferio izquierdo en su extremo inicial, y al hemisferio derecho en su extremo final.
  - R-L: Clusters ínter-hemisferio derecha-izquierda. Opuestos a los L-R.
  - R-R: Clusters intra-hemisferio izquierdo, es decir, sus fibras intersectan el hemisferio derecho en ambos extremos.
  
La clasificación de los clusters se realiza en base a los perfiles de conectividad según las intersecciones obtenidas con el código mencionado anteriormente. El detalle teórico se encuentra en mi MT, además, el código en Python está comentado y explicado. 

### Ejemplo de uso
Este script no fue escrito considerando ejecución desde la terminal, por lo que las rutas y variables relevantes deben ser editadas directamente en el código fuente, y ejecutar desde ahí (por ejemplo con Spyder).

### Archivos de entrada
Referirse al código comentado para ver los archivos necesarios para el funcionamiento del script.

### Archivos de salida
- Se creará una carpeta llamada _bundle_classification_, en donde para cada sujeto se crearán 4 carpetas correspondientes a las 4 categorías (L-L, L-R, R-L, R-R), y se copiarán los clusters a dichas carpetas según la clase que determine el algoritmo. Además, se creará una carpeta adicional _Discarded_, en donde se copiarán aquellos clusters que no corresponden a ninguna de las otras categorías.

## Intersection Classification <a name="intersection_classification"></a>
Este algoritmo realiza post-procesamiento y obtiene los datos de intersección finales para cada clúster. Esto es necesario ya que como la intersección inicial se realiza para cuatro escenarios posibles distintos, tenemos 4 archivos distintos de intersección para cada cluster. Por lo tanto, luego de haber clasificado cada cluster, este código:

  - Filtra las fibras que ya no corresponda considerar (por ejemplo, si se determinó que un fascículo es definitivamente L-L, entonces se eliminan todas las fibras potencialmente ínter-hemisferio que se encontraron inicialmente), y entrega sólo un archivo .intersectiondata por cada cluster, según la categoría a la cual pertenezca (L-L, L-R, R-L, R-R).
  
  - Integra los datos de intersección en el caso de clusters ínter-hemisferio; es decir, toma las intersecciones finales del extremo inicial (que tenían un índice 1000000) y las reemplaza con las intersecciones iniciales del cluster en el sentido opuesto, eliminando así todas las intersecciones "dummy" con valor de 1000000.
  
Para más detalle referirse al código comentado.

### Ejemplo de uso
Este script no fue escrito considerando ejecución desde la terminal, por lo que las rutas y variables relevantes deben ser editadas directamente en el código fuente, y ejecutar desde ahí (por ejemplo con Spyder).

### Archivos de entrada
Referirse al código comentado para ver los archivos necesarios para el funcionamiento del script.

### Archivos de salida
- Carpeta "/intersection/sub/Final", y 3 sub-carpetas:
  - L-L: Clusters intra-hemisferio izquierdo.
  - R-R: Clusters intra-hemisferio derecho.
  - Inter: Clusters ínter-hemisferio.

- Cada una de estas carpetas guarda un archivo .intersectiondata post-procesado, según la categoría asignada al cluster anteriormente por _bundle_classification_.

## Referencias
<a id="1">[1]</a>
F. Silva, M. Guevara, C. Poupon, J.-F. Mangin, C. Hernandez, and P. Guevara, “Cortical Surface Parcellation Based on Graph Representation of Short Fiber Bundle Connections,” in 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), 2019.
