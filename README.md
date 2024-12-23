# TGVD-Proyecto-Interseccion


Este proyecto implementa un módulo en C++ con Python mediante `pybind11` que permite realizar consultas eficientes de intersección entre tractografía cerebral y mallas corticales. El código incluye algoritmos como el teorema del eje separador (SAT) y el algoritmo de Möller-Trumbore para intersección rayo-triángulo, utilizando una estructura Octree para mejorar la eficiencia.

## Contenido

1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Dependencias](#dependencias)
3. [Compilación e Instalación](#compilación-e-instalación)
4. [Datos de Entrada](#datos-de-entrada)
5. [Uso](#uso)
6. [Resultados](#resultados)
7. [Referencias](#referencias)

## Estructura del Proyecto

```
├── build/                    # Carpeta de compilación
├── figures/                  # Gráficas generadas
├── gt/                       # Datos ground truth
├── intersection/             # Datos de intersección generados
├── meshes/                   # Mallas corticales
├── results/                  # Resultados generados
├── test/                     # Subconjuntos de tractografía
├── tract/                    # Datos de tractografía
├── bundleTools.py            # Utilidades para manejo de tractografía
├── bundleTools3.py           # Segunda versión de las utilidades
├── intersection_utils.py     # Funciones de intersección
├── main.py                   # Script principal
├── octree.cpp                # Implementación del Octree
├── queries.py                # Consultas relacionadas con intersección
├── CMakeLists.txt            # Configuración de compilación con CMake
└── README.md                 # Documentación del proyecto
```

## Dependencias

Antes de compilar e instalar este proyecto, asegúrese de tener las siguientes dependencias instaladas:

1. Python 3.8 o superior
2. pybind11
3. OpenMP
4. NumPy
5. Matplotlib
6. TQDM
7. FURY

### Instalación de dependencias
Puede instalar las dependencias de Python mediante `pip`:

```bash
pip install numpy matplotlib tqdm fury
```

Para pybind11 y OpenMP:

- En Ubuntu:
  ```bash
  sudo apt install libomp-dev
  sudo apt install pybind11-dev
  ```

- En Windows, instale `pybind11` desde el gestor de paquetes de su compilador.

## Compilación e Instalación

1. **Configurar CMake**
   En el directorio del proyecto, ejecute:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Generar el módulo**
   Esto generará el archivo `octree_module.cpython*.so` en el directorio `build/`.

3. **Pruebas**
   Desde la carpeta base, ejecute:

   ```bash
   python main.py
   ```

## Datos de Entrada

Los datos necesarios para este proyecto están organizados en las siguientes carpetas:

1. **Mallas Corticales:** Ubicadas en `meshes/`, contienen modelos 3D en formato `.obj`.
2. **Tractografía Cerebral:** Ubicadas en `tract/`, contienen datos en formato `.bundles`.

Los datos de tractografía utilizados están disponibles en el siguiente enlace:
[Datos de Tractografía](https://drive.google.com/drive/folders/1rqpFk3GRi0x7Xu8bCJKDWgMd8aJXFWr_?usp=sharing)

## Uso

El script principal `main.py` permite realizar pruebas y evaluaciones utilizando las implementaciones disponibles. A continuación, se detalla un ejemplo de uso:

```bash
python main.py
```

### Flujo General del Script:
1. Cargar las mallas corticales y tractografía.
2. Construir el Octree y subdividir las mallas.
3. Realizar consultas de intersección entre fibras y triángulos de la malla.
4. Generar resultados y gráficas de rendimiento.

### Ejemplo de Configuración
Modifique las rutas y parámetros en `main.py` para personalizar las pruebas:

```python
mesh_lh_path = 'meshes/lh.obj'
bundles_path = 'tract/3Msift_t_MNI_21p.bundles'
```

## Resultados

El proyecto genera varios resultados:

1. **Tiempo de Construcción del Octree:**
   - Comparación entre la subdivisión estándar y la basada en vértices.
   - Guardado en `figures/octree_construction_time.png`.

2. **Intersección entre Fibras y Mallas:**
   - Resultados guardados en `results/`.

3. **Métricas de Evaluación:**
   - Recall y discrepancia entre diferentes métodos.

4. **Gráficas de Desempeño:**
   - Comparación del tiempo de ejecución entre Octree y métodos de fuerza bruta.

## Referencias

- "Real-Time Collision Detection", Christer Ericson.
- [Google Drive: Datos de Tractografía](https://drive.google.com/drive/folders/1rqpFk3GRi0x7Xu8bCJKDWgMd8aJXFWr_?usp=sharing)

---
Cualquier consulta o problema, por favor contactar al desarrollador del proyecto.

