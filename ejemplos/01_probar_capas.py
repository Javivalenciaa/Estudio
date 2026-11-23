# =============================================================================
# ejemplos/01_probar_capas.py -> mira por dentro de Lineal, ReLU y softmax
# =============================================================================
# CÓMO EJECUTARLO (desde la carpeta principal del repo, NO desde ejemplos/):
#       python -m ejemplos.01_probar_capas      <- en algunos sistemas da error
#       python ejemplos/01_probar_capas.py      <- este funciona siempre
#
# Cada bloque imprime algo y te digo debajo qué deberías ver.

import os, sys

# [PYTHON] Esto añade la carpeta principal del repo a la lista donde Python busca
#   código, para que "import nn" funcione aunque ejecutes el archivo desde otro
#   sitio. __file__ es la ruta de este archivo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nn.capas import Lineal, ReLU
from nn.perdidas import softmax, entropia_cruzada

# [PYTHON] Pide a NumPy que imprima con 3 decimales y sin notación científica.
np.set_printoptions(precision=3, suppress=True)

# ---------------------------------------------------------------------------
print("=== 1) Formas de una capa Lineal(40, 64) con 5 canciones ===")
capa = Lineal(40, 64)
X = np.random.default_rng(1).standard_normal((5, 40))  # 5 canciones, 40 rasgos
Z = capa.forward(X)
print("X:", X.shape, " W:", capa.W.shape, " b:", capa.b.shape, " Z:", Z.shape)
# DEBE SALIR: X: (5, 40)  W: (40, 64)  b: (64,)  Z: (5, 64)

# ---------------------------------------------------------------------------
print("\n=== 2) ReLU: lo negativo pasa a 0 ===")
A = np.array([[-2.0, 0.5, 3.0, -0.1]])
print("entrada:", A)
print("salida :", ReLU().forward(A))
# DEBE SALIR: [[-2.   0.5  3.  -0.1]] -> [[0.  0.5 3.  0. ]]   (puede aparecer -0.)

# ---------------------------------------------------------------------------
print("\n=== 3) Softmax: puntuaciones -> probabilidades ===")
Zs = np.array([[1.0, 0.4, -0.6]])
P = softmax(Zs)
print("probabilidades:", P, " suma:", P.sum())
# DEBE SALIR: [[0.571 0.313 0.115]]  suma: 1.0

# ---------------------------------------------------------------------------
print("\n=== 4) Entropía cruzada: la nota de error y la culpa dZ ===")
perdida, dZ = entropia_cruzada(Zs, np.array([1]))  # el género correcto es el 1
print("pérdida:", round(perdida, 3), " dZ:", dZ)
# DEBE SALIR: pérdida: 1.16   dZ: [[ 0.571 -0.687  0.115]]
# Fíjate: en el género correcto (posición 1) dZ es NEGATIVO ("sube esta
# puntuación") y en los demás POSITIVO ("bájalas").

# ---------------------------------------------------------------------------
print("\n=== 5) Backward de Lineal: las formas de las pendientes ===")
capa.forward(X)
dX = capa.backward(np.ones((5, 64)))
print("dW:", capa.dW.shape, " db:", capa.db.shape, " dX:", dX.shape)
# DEBE SALIR: dW: (40, 64)  db: (64,)  dX: (5, 40)
# Regla: la pendiente de algo SIEMPRE tiene la forma de ese algo.
