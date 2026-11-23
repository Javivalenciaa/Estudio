# 🎵 Clasificador de género musical — Parte de la Persona A (la red neuronal)

Este repositorio contiene **solo la parte A** de nuestro proyecto de Aprendizaje Automático: **la red neuronal escrita a mano con NumPy**, con cada línea explicada para alguien que sabe poco de Python y poco de modelos.

> Objetivo del repo: que leyendo los archivos **en orden** entiendas (1) qué hace el modelo, (2) por qué se hace cada cosa y (3) cómo se escribe en Python.

---

## 1. ¿Qué hace el proyecto?

Le damos a un programa una canción y nos dice de qué **género** es (rock, jazz, clásica, hip-hop…). Usamos el dataset **GTZAN**: 1000 canciones de 30 s, 10 géneros.

Es un problema de **clasificación multiclase**: dado un dato, elegir **una categoría entre 10**.

```
canción (audio) ──► 40 números (rasgos) ──► RED NEURONAL ──► 10 probabilidades ──► género ganador
```

### ¿Qué son los 40 números?
Una canción son miles de muestras de sonido: demasiado para meterlas tal cual. Se resume con **MFCC** (*Mel-Frequency Cepstral Coefficients*): números que describen el **timbre** del sonido (si suena grave, brillante, áspero…) de forma parecida a como lo oye el oído humano. Se calculan 20 coeficientes y, de cada uno, se guarda su **media** y su **desviación típica** a lo largo de la canción: 20 × 2 = **40 rasgos por canción**. De sacarlos se encarga la Persona C con la librería `librosa`.

Así, el conjunto de datos es una tabla **X** de 1000 filas (canciones) × 40 columnas (rasgos), y un vector **y** con el género (0–9) de cada canción.

---

## 2. Reparto del grupo

| Persona | Parte | Qué hace |
|---|---|---|
| **A (este repo)** | **La red** | Capas (`Lineal`, `ReLU`), softmax + entropía cruzada, la clase `Red`, guardar/cargar, *gradient check* |
| B | Entrenamiento | Optimizadores (SGD, Momentum, Adam), bucle de entrenamiento, regularización L2, parada temprana |
| C | Datos y extras | Extraer MFCC, normalizar, dividir train/val/test, métricas, comparar con un modelo lineal, (opcional) k-means, web con Streamlit |

**Cómo encajan:** B llama a tu `red.forward(X)`, calcula la pérdida con tu `entropia_cruzada`, llama a tu `red.backward(dZ)` y luego usa `red.params()` para **girar los botones** con su optimizador. Por eso tu interfaz tiene que ser exactamente esa (ver sección 6).

---

## 3. La red, explicada para tontos

Una red neuronal es una **máquina con botones**:

- Entran 40 números (la canción).
- Pasan por varias **capas**. Cada capa hace una cuenta con sus botones (los **pesos**).
- Salen 10 puntuaciones, una por género.
- **Entrenar** = ajustar los botones poco a poco para que la máquina acierte más.

Nuestra arquitectura:

```
 40 rasgos ─► [Lineal 40→64] ─► ReLU ─► [Lineal 64→32] ─► ReLU ─► [Lineal 32→10] ─► 10 puntuaciones
                                                                                      │
                                                                       softmax ───────┘ ─► 10 probabilidades
```

Número de botones (parámetros): 40·64+64 = **2624**, 64·32+32 = **2080**, 32·10+10 = **330** → **5034** en total.

### Las piezas

| Pieza | Qué hace | Archivo | Temario |
|---|---|---|---|
| `Lineal` | `Z = X @ W + b` | `nn/capas.py` | Regresión lineal (con matriz W en vez de un vector) |
| `ReLU` | pone a 0 lo negativo; introduce no linealidad | `nn/capas.py` | Funciones base / redes neuronales |
| `softmax` | puntuaciones → probabilidades que suman 1 | `nn/perdidas.py` | Clasificación multiclase |
| `entropia_cruzada` | nota de error `−log(prob. del género correcto)` | `nn/perdidas.py` | Perspectiva probabilística / máxima verosimilitud |
| `backward` | reparte la "culpa" del error entre los botones | `nn/capas.py`, `nn/red.py` | Regla de la cadena / gradientes |
| `Red` | encadena capas | `nn/red.py` | — |
| `comprobar_gradientes` | verifica que el backward está bien | `nn/gradcheck.py` | Derivada como límite |

---

## 4. Ideas clave (con el porqué)

### 4.1 Forward: pasar las canciones por la red
Se procesan **todas las canciones a la vez** con multiplicación de matrices (`@`). Con `X` de forma `(N, 40)` y `W` de forma `(40, 64)`, `X @ W` da `(N, 64)`: una fila por canción y una columna por neurona. Es lo mismo que hacer N cuentas separadas, pero muchísimo más rápido.

### 4.2 Softmax y entropía cruzada
- La red da puntuaciones libres. **Softmax** las convierte en probabilidades: `p_i = e^{z_i} / Σ e^{z_j}`. *No es lo mismo que la sigmoide*: la sigmoide sirve para **2 clases** (una probabilidad); softmax generaliza a **C clases** y las probabilidades compiten entre sí.
- La **entropía cruzada** castiga más cuanto menos probabilidad dé la red al género correcto. Prob. 0.9 → pérdida 0.105; prob. 0.1 → 2.303.
- Truco numérico: se resta el máximo de cada fila antes de `exp` para evitar desbordamientos.
- La combinación softmax + entropía cruzada tiene un gradiente precioso: **`dZ = (P − Y) / N`**.

### 4.3 Backpropagation (la regla de la cadena)
Si el error depende de Z y Z depende de W, entonces `dError/dW = dError/dZ · dZ/dW`. Aplicado capa a capa, **de atrás hacia delante**:

```
dW  = X.T @ dZ        # culpa de los pesos
db  = dZ.sum(axis=0)  # culpa de los sesgos
dX  = dZ @ W.T        # culpa que se pasa a la capa anterior
```
Truco para no liarse con las `.T`: **el gradiente de algo siempre tiene la forma de ese algo**.

### 4.4 ¿Por qué ReLU?
Sin ReLU, apilar capas lineales equivale a **una sola** capa lineal. ReLU introduce el "doblez" que permite fronteras no rectas.

### 4.5 Inicialización de He
`W = aleatorio · √(2/n_in)`. Mantiene el tamaño de los números estable al atravesar capas con ReLU.

### 4.6 Gradient check
Compara los gradientes de tu `backward` con el cálculo numérico `(L(w+ε) − L(w−ε)) / 2ε`. Es la forma de **demostrar** que tu backprop es correcto. Debe dar error relativo < 1e-6 (nosotros: ~1e-10).

---

## 5. ¿Es un k-means o un modelo lineal?

- **Clasificador multiclase (nuestra red)**: *supervisado* — aprende con etiquetas (sabemos el género de cada canción de entrenamiento).
- **k-means**: *no supervisado* — agrupa sin etiquetas; el nº de clusters no tiene por qué coincidir con el de clases y sus grupos no tienen nombre.
- **Línea base**: la **regresión softmax** (la misma red **sin capas ocultas**: `crear_mlp(40, [], 10)`, 410 parámetros). Sirve para comprobar si la red profunda realmente aporta algo.

---

## 6. La interfaz que B y C necesitan de ti

```python
from nn.red import crear_mlp, guardar, cargar_pesos
from nn.perdidas import entropia_cruzada, softmax

red = crear_mlp(40, [64, 32], 10, semilla=0)   # montar la red

Z = red.forward(X)                  # (N, 10) puntuaciones
perdida, dZ = entropia_cruzada(Z, y)  # nota de error + culpa inicial
red.backward(dZ)                    # rellena dW y db en cada capa
for p, g in red.params():           # lista de (botón, pendiente)
    p -= lr * g                     # (esto lo hará B con su optimizador)

red.predecir(X)         # (N,) género elegido
red.predecir_proba(X)   # (N, 10) probabilidades
guardar(red, "modelo.npz", mu=mu, sigma=sigma)
cargar_pesos(red, "modelo.npz")
```

`params()` devuelve **los propios arrays** (no copias), así que modificarlos en el sitio (`p -= ...`) cambia de verdad la red.

---

## 7. Estructura del repo y orden de lectura

```
docs/PYTHON_PARA_TONTOS.md      ← LEE ESTO PRIMERO si Python te suena a chino
nn/capas.py                     ← 1º  Lineal y ReLU
nn/perdidas.py                  ← 2º  softmax y entropía cruzada
nn/red.py                       ← 3º  Red, crear_mlp, guardar/cargar
nn/gradcheck.py                 ← 4º  detector de errores
ejemplos/01_probar_capas.py     ← ejecútalo: ver las formas y los números
ejemplos/02_gradcheck_y_memorizar.py  ← ejecútalo: pruebas de que todo va bien
ejemplos/datos_falsos.py        ← canciones inventadas para probar
```

## 8. Cómo ejecutarlo

```bash
# 1) (opcional pero recomendable) entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux

# 2) instalar NumPy
pip install -r requirements.txt

# 3) ejecutar los ejemplos DESDE LA CARPETA PRINCIPAL del repo
python ejemplos/01_probar_capas.py
python ejemplos/02_gradcheck_y_memorizar.py
```

⚠️ Ejecuta siempre desde la carpeta principal, **no** desde dentro de `nn/`.

---

## 9. Preguntas típicas del profesor (y respuesta corta)

- **¿Por qué entropía cruzada y no error cuadrático?** Porque la salida es una distribución de probabilidad sobre categorías; la entropía cruzada es la log-verosimilitud negativa y con softmax da el gradiente `P − Y`.
- **¿Qué hace backprop?** Aplica la regla de la cadena de atrás hacia delante, reutilizando lo calculado en el forward, para obtener el gradiente de la pérdida respecto a cada parámetro.
- **¿Cómo sabes que tu gradiente es correcto?** Gradient check con diferencias finitas centradas: error relativo ≈ 1e-10.
- **¿Por qué restas el máximo en softmax?** Para evitar overflow en `exp`; no cambia el resultado.
- **¿Por qué no basta con capas lineales?** Su composición sigue siendo lineal; ReLU aporta no linealidad.
- **¿Softmax es la sigmoide?** La sigmoide es el caso de 2 clases; softmax con 2 clases equivale a ella.
- **¿Qué pasa con 1000 × 40?** Es diminuto para un ordenador: la matriz ocupa ~320 KB.

## 10. Créditos
Temario: *The JuanBook of Machine Learning* (J. Maroñas). Dataset: GTZAN.
