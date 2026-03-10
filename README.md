# Taller Práctico 2 — Múltiples Filtros de Imágenes Aplicados para Procesamiento de Video con CUDA

**Autor:** Carlos Alberto Cardona Pulido  
**Curso:** Arquitectura de Computadores / Computación de Alto Rendimiento  
**Universidad:** Universidad Sergio Arboleda  
**Fecha:** Marzo 2026

---

## Descripción General

Este laboratorio implementa cinco filtros de procesamiento de imágenes sobre video utilizando la tecnología **CUDA** a través de la librería **Numba** en Python. La arquitectura adoptada sigue tres etapas secuenciales:

1. **Split:** El video de entrada se descompone en fotogramas individuales almacenados en disco, evitando así saturar la memoria RAM de la instancia de Google Colab.
2. **Process:** Cada fotograma es procesado de forma independiente mediante un kernel CUDA que paraleliza el trabajo a nivel de píxel.
3. **Join:** Los fotogramas procesados se reensamblan en un archivo de video `.mp4` codificado en H.264 para garantizar su reproducción en cualquier navegador.

Para cada filtro se implementó adicionalmente una versión equivalente en **CPU**, con el propósito de comparar los tiempos de ejecución y calcular el **Speedup** obtenido por el uso de la GPU.

---

## Entorno de Ejecución

| Parámetro | Valor |
|---|---|
| Plataforma | Google Colab |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| CPU | Intel Xeon (2 vCPUs) |
| Framework CUDA | Numba (`numba-cuda`) |
| Lenguaje | Python 3.12 |
| Resolución de video de prueba | 1920 × 1080 px @ 30 fps |

---

## Filtros Implementados

### 1. Inversión de Colores (Negativo)

**Descripción teórica**

La inversión de colores es una operación de procesamiento **puntual**: el nuevo valor de cada píxel depende exclusivamente de su valor original, sin considerar píxeles vecinos. En el modelo RGB de 8 bits por canal (rango [0, 255]), la fórmula aplicada es:

```
R_out = 255 - R_in
G_out = 255 - G_in
B_out = 255 - B_in
```

Este proceso simula el comportamiento de los negativos fotográficos analógicos: los tonos claros se convierten en oscuros y viceversa.

**Implementación CUDA**

Cada hilo del kernel procesa exactamente un píxel de forma completamente independiente. Dado que no existe ninguna dependencia entre píxeles, este filtro presenta el máximo potencial de paralelización posible. La malla de hilos se configura en bloques de 16×16, cubriendo la totalidad de la imagen.

**Análisis de rendimiento**

| Métrica | GPU | CPU (NumPy) |
|---|---|---|
| Tiempo total (322 frames) | ~1.8 s | ~2.1 s |
| Tiempo por frame | ~5.6 ms | ~6.5 ms |
| Speedup | ~1.2× | — |

El Speedup en este filtro es modesto porque la operación CPU vectorizada con NumPy también es muy eficiente. Sin embargo, la ventaja de la GPU se amplifica considerablemente al escalar a videos de mayor resolución o duración.

---

### 2. Blur (Desenfoque por Media — Box Blur)

**Descripción teórica**

El desenfoque por media es una operación de **vecindad**: el valor de cada píxel de salida se calcula como el promedio aritmético de todos los píxeles dentro de una ventana cuadrada de tamaño (2r+1)×(2r+1), donde r es el radio del kernel:

```
P_out(i,j) = 1 / (2r+1)² × Σ P_in(i+m, j+n)    con m,n ∈ [-r, r]
```

A diferencia del negativo, este filtro requiere acceso a píxeles adyacentes, lo que introduce dependencias de memoria que deben gestionarse cuidadosamente.

**Implementación CUDA con Memoria Compartida**

Para radio r=3 (ventana 7×7), cada píxel requiere 49 lecturas. Si todas se realizan desde **memoria global** (latencia ~600 ciclos), el costo es prohibitivo para un video de alta resolución. Por este motivo se utilizó **memoria compartida** (*shared memory*, latencia ~4 ciclos):

- Cada bloque de hilos (16×16) carga su tile más un halo de `r` píxeles en cada borde en memoria compartida.
- El tile en shared memory tiene dimensiones fijas de 24×24×3 (constante de compilación requerida por Numba).
- Tras sincronizar con `cuda.syncthreads()`, cada hilo lee sus 49 vecinos desde shared memory en lugar de memoria global, reduciendo los accesos globales de O(n·k²) a O(n).

**Implementación CPU**

Se empleó `scipy.ndimage.uniform_filter`, que internamente utiliza una implementación en C altamente optimizada, permitiendo una comparación de tiempos justa sobre la totalidad de los frames.

**Análisis de rendimiento**

| Métrica | GPU (shared mem) | CPU (scipy) |
|---|---|---|
| Tiempo total (322 frames, r=3) | ~18 s | ~95 s |
| Tiempo por frame | ~56 ms | ~295 ms |
| Speedup | ~5.3× | — |

La memoria compartida es determinante: sin ella, el kernel con r=3 sobre 1080p superaba los 20 minutos de ejecución. Con shared memory, el tiempo cae a menos de 20 segundos, confirmando que la gestión de la jerarquía de memoria es tan importante como el paralelismo en sí.

---

### 3. Recorte de Imagen (Crop)

**Descripción teórica**

El recorte consiste en extraer una subregión rectangular de la imagen original, definida por una coordenada de inicio (x_start, y_start) y las dimensiones (width_crop × height_crop). Desde el punto de vista computacional es una **copia selectiva de memoria** basada en un desplazamiento:

```
P_dest(i, j) = P_src(i + x_start, j + y_start)
```

**Implementación CUDA**

La malla de hilos se dimensiona según la imagen **destino** (pequeña), no la fuente. Cada hilo calcula su posición en la imagen fuente sumando el offset. Se incluye validación de límites: si las coordenadas mapeadas exceden las dimensiones originales, el píxel se asigna a 0 (negro) para evitar accesos a memoria no asignada en la GPU.

**Implementación CPU**

Se utilizó el slicing nativo de NumPy (`img[x:x+h, y:y+w]`), que realiza una copia contigua de memoria extremadamente eficiente.

**Análisis de rendimiento**

| Métrica | GPU | CPU (NumPy slice) |
|---|---|---|
| Tiempo total (322 frames) | ~2.5 s | ~1.4 s |
| Tiempo por frame | ~7.8 ms | ~4.3 ms |
| Speedup | ~0.56× | — |

En este filtro la CPU supera a la GPU. Esto se debe a que NumPy ejecuta la copia de memoria con rutinas BLAS altamente optimizadas, mientras que la GPU incurre en overhead de transferencia CPU↔GPU para una operación que no tiene carga computacional significativa. Este resultado ilustra que CUDA no es siempre la herramienta óptima: para operaciones de copia simple, la CPU puede ser más eficiente.

---

### 4. Imagen Binaria (Binarización por Umbral)

**Descripción teórica**

La binarización transforma cada píxel en uno de dos valores posibles: 0 (negro) o 255 (blanco), según si su intensidad media supera un umbral T:

```
Intensidad = (R + G + B) / 3

P_out(i,j) = 255   si Intensidad > T
P_out(i,j) = 0     si Intensidad ≤ T
```

El umbral T=128 representa el punto medio del rango [0, 255]. Valores bajos de T producen imágenes predominantemente blancas; valores altos, predominantemente negras.

**Implementación CUDA**

Al igual que el negativo, es una operación puntual: cada hilo calcula la intensidad de su píxel y aplica la comparación con el umbral. La imagen de salida es de un solo canal (escala de grises binaria), que se convierte a BGR de 3 canales antes de guardarse para compatibilidad con OpenCV y el VideoWriter.

**Análisis de rendimiento**

| Métrica | GPU | CPU (NumPy vectorizado) |
|---|---|---|
| Tiempo total (322 frames) | ~2.0 s | ~2.3 s |
| Tiempo por frame | ~6.2 ms | ~7.1 ms |
| Speedup | ~1.15× | — |

El comportamiento es similar al del negativo: ambas son operaciones puntuales donde NumPy vectorizado compite directamente con CUDA para resoluciones moderadas.

---

### 5. Detección de Bordes — Operador Sobel

**Descripción teórica**

La detección de bordes identifica los puntos de cambio brusco de intensidad en una imagen, los cuales corresponden a los límites de los objetos. El operador Sobel estima el gradiente de la imagen en dos direcciones usando convolución con kernels 3×3 predefinidos:

**Kernel horizontal Kx** (detecta bordes verticales):
```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

**Kernel vertical Ky** (detecta bordes horizontales):
```
[-1 -2 -1]
[ 0  0  0]
[ 1  2  1]
```

La magnitud del gradiente, que representa la fuerza del borde en cada píxel, se obtiene como:

```
G = sqrt(Gx² + Gy²)
```

**Implementación CUDA**

El proceso consta de dos etapas: primero se convierte el frame a escala de grises en CPU (ponderación estándar de luminancia: 0.2989·R + 0.5870·G + 0.1140·B), y luego cada hilo CUDA calcula las convoluciones Gx y Gy para su píxel usando los 8 vecinos de la ventana 3×3. La magnitud se calcula con `cuda.libdevice.sqrtf`, la función de punto flotante nativa de CUDA. Los píxeles del borde de la imagen (sin vecinos completos) se asignan a 0.

**Implementación CPU**

Se implementaron los mismos bucles de convolución explícita en Python puro, equivalentes al kernel CUDA, para medir el costo real de la operación secuencial.

**Análisis de rendimiento**

| Métrica | GPU | CPU (bucles Python) |
|---|---|---|
| Tiempo total (322 frames) | ~22 s | ~3200 s (estimado) |
| Tiempo por frame | ~68 ms | ~9940 ms |
| Speedup | **~146×** | — |

Este filtro exhibe el mayor Speedup del taller. La doble convolución 3×3 sobre una imagen de 1080p implica ~6.2 millones de operaciones por frame en la CPU secuencial. La GPU los ejecuta todos en paralelo en decenas de milisegundos, haciendo que la medición CPU se realice sobre una muestra de 5 frames y se extrapole al total.

---

## Comparativa General de Speedup

| Filtro | Speedup GPU/CPU | Tipo de operación |
|---|---|---|
| Inversión de colores | ~1.2× | Puntual |
| Blur (r=3, shared memory) | ~5.3× | Vecindad (memoria compartida) |
| Recorte | ~0.56× | Copia de memoria |
| Binarización | ~1.15× | Puntual |
| Sobel | ~146× | Convolución doble |

---

## Conclusiones

1. **El paralelismo masivo de CUDA es más efectivo cuanto mayor es la carga computacional por píxel.** Los filtros puntuales (negativo, binario) muestran Speedups modestos frente a NumPy vectorizado, mientras que filtros de vecindad como Sobel alcanzan Speedups de dos órdenes de magnitud.

2. **La jerarquía de memoria es tan crítica como el paralelismo.** El filtro de blur sin memoria compartida era impracticable (>20 minutos por video). La adopción de *shared memory* redujo el tiempo a menos de 20 segundos, demostrando que el acceso eficiente a memoria puede ser el cuello de botella dominante en kernels CUDA.

3. **CUDA no es universalmente superior.** El recorte de imagen muestra que para operaciones de copia simple, la implementación CPU con NumPy supera a la GPU, cuyo overhead de transferencia de datos domina sobre el trabajo computacional.

4. **La arquitectura Split → Process → Join es escalable.** Al persistir los frames en disco en lugar de mantenerlos en RAM, es posible procesar videos de alta resolución y larga duración sin limitaciones de memoria, lo cual es fundamental para aplicaciones reales de visión computacional.

5. **El re-encodeo a H.264 con ffmpeg es indispensable** para la reproducción inline en navegadores, ya que el codec `mp4v` generado por OpenCV no es soportado por los players HTML5 modernos.

