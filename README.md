# TGVD-Proyecto-Intersección

Proyecto Final para el curso **Tópicos en Manejo de Grandes Volúmenes de Datos**. Este proyecto implementa un módulo en C++ con Python mediante `pybind11` que permite realizar consultas eficientes de intersección entre tractografía cerebral y mallas corticales. Utiliza algoritmos como el **Teorema del Eje Separador (SAT)** y el **Algoritmo de Möller-Trumbore** para calcular intersecciones rayo-triángulo, optimizando la eficiencia mediante una estructura Octree.

## Requisitos Previos

Para compilar y ejecutar este proyecto, asegúrate de contar con las siguientes herramientas instaladas:

1. **Sistema Operativo:** Linux
2. **Compilador:**
   - `cmake` (>= 3.11)
   - `g++` (>= 11.0) con soporte para C++17
3. **Librerías:**
   - `pybind11`
   - `OpenMP`

Adicionalmente, es necesario un entorno de Python con las siguientes dependencias:

- **Python:** 3.9 o superior
- **Paquetes Python:** `pybind11`, `numpy`, `matplotlib`, `seaborn`, `nilearn`, `vtk`, `tqdm`, `fury`

---

## Instalación del Entorno

Recomendamos crear un entorno virtual para una configuración limpia y reproducible.

### Crear y Activar un Nuevo Entorno

1. **Crear el entorno virtual `tgvd_env`:**
   ```bash
   python3.9 -m venv tgvd_env
   ```

2. **Activar el entorno:**
   - **Linux/Mac:**
     ```bash
     source tgvd_env/bin/activate
     ```
   - **Windows:**
     ```cmd
     tgvd_env\Scripts\activate
     ```

3. **Actualizar `pip` y `setuptools`:**
   ```bash
   pip install --upgrade pip setuptools
   ```

### Instalar Dependencias

Ejecuta el siguiente comando para instalar las dependencias de Python:

```bash
pip install pybind11 numpy matplotlib seaborn nilearn vtk tqdm fury
```

Adicionalmente, instala `pybind11` y `OpenMP` en tu sistema:

- **Ubuntu:**
  ```bash
  sudo apt install libomp-dev pybind11-dev
  ```

---

## Descarga y Organización de los Datos

### Descargar Datos

Para ejecutar correctamente este proyecto, descarga los datos necesarios desde el siguiente enlace:

[**Datos de Tractografía y Mallas Corticales**](https://drive.google.com/drive/folders/1rqpFk3GRi0x7Xu8bCJKDWgMd8aJXFWr_?usp=sharing)

### Organización de Archivos

Una vez descargados, organiza los datos de la siguiente forma en el directorio raíz del proyecto:

```plaintext
TGVD-Proyecto-Interseccion/
├── meshes/                  # Mallas corticales (ya incluidas)
├── tract/                   # Carpeta donde debes colocar los datos descargados
    ├── 3Msift_t_MNI_21p.bundles
    └── 3Msift_t_MNI_21p.bundlesdata
```

---

## Instrucciones de Instalación y Ejecución

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/jorvergara/TGVD-Proyecto-Interseccion.git
   cd TGVD-Proyecto-Interseccion
   ```

2. **Crea y compila el proyecto:**
   ```bash
   chmod +x build.sh
   ./build.sh
   ./compile.sh
   ```

3. **Ejecuta el proyecto:**
   ```bash
   ./run.sh
   ```

---

## Pruebas

Para realizar pruebas, ejecuta el archivo principal:

```bash
python main.py
```

El script principal incluye comparaciones entre la implementación de Octree y métodos tradicionales de cálculo de intersecciones.

---

## Referencias

- **Real-Time Collision Detection**, Christer Ericson.
- [Google Drive: Datos de Tractografía](https://drive.google.com/drive/folders/1rqpFk3GRi0x7Xu8bCJKDWgMd8aJXFWr_?usp=sharing)

---

## Autor

- Jorge Vergara

--- 

Este README ahora está mejor estructurado, incluye todas las instrucciones necesarias y es fácil de seguir.