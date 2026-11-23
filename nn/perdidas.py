# =============================================================================
# nn/perdidas.py  ->  SOFTMAX Y ENTROPÍA CRUZADA (la "nota de error")
# =============================================================================
#
# QUÉ HAY EN ESTE ARCHIVO
#   1) softmax          : convierte puntuaciones en PROBABILIDADES (suman 100 %).
#   2) entropia_cruzada : mide cuánto se ha equivocado la red (un solo número)
#                         y calcula la "culpa" inicial que arranca el backward.
#
# [PYTHON] Aquí NO hay clases, solo FUNCIONES, porque no tienen botones propios
#   que recordar: reciben datos, calculan y devuelven un resultado.
#
# [TEMARIO] Esto es el capítulo de CLASIFICACIÓN MULTICLASE: softmax + entropía
#   cruzada (la generalización a varias clases de la sigmoide + entropía
#   cruzada binaria de la clasificación binaria).
# =============================================================================

import numpy as np


# =============================================================================
# softmax
# =============================================================================
# [MODELO] La red devuelve, por cada canción, 10 PUNTUACIONES (una por género),
#   por ejemplo [1.0, 0.4, -0.6, ...]. Son difíciles de interpretar: pueden ser
#   negativas, grandes, etc. Softmax las convierte en PROBABILIDADES: números
#   entre 0 y 1 que suman 1 (o sea, 100 %).
#
#   Fórmula para cada puntuación z_i:   p_i = e^(z_i) / (suma de todas las e^(z_j))
#
#   Ejemplo con z = [1.0, 0.4, -0.6]:
#       e^1.0 = 2.718   e^0.4 = 1.492   e^-0.6 = 0.549     suma = 4.759
#       probabilidades = [2.718/4.759, 1.492/4.759, 0.549/4.759]
#                      = [0.571, 0.313, 0.115]   (suman 1 -> casi 1 con redondeo)
#   La puntuación más alta se lleva la probabilidad más alta.
#
# [IMPORTANTE] Softmax NO es "element-wise" (casilla a casilla): cada salida
#   depende de TODAS las puntuaciones de la fila (por el denominador, la suma).
def softmax(Z):
    """Convierte cada fila de puntuaciones en probabilidades que suman 1.

    Entrada Z: tabla (N, C) -> N canciones, C géneros.
    Salida   : tabla (N, C) de probabilidades.
    """
    # --- Truco de estabilidad numérica ---
    # [MODELO] e^1000 es un número tan enorme que el ordenador lo da como
    #   "infinito" y todo se rompe (salen "nan"). Pero si a TODAS las
    #   puntuaciones de una fila les restas el mismo número, el resultado de
    #   softmax NO cambia (arriba y abajo se cancela). Así que restamos el
    #   MAYOR de cada fila: el mayor pasa a ser 0 y el resto negativos, y
    #   e^(negativo) nunca se desborda.
    # [PYTHON] Z.max(axis=1) = el mayor de cada FILA (axis=1 recorre las columnas).
    #   keepdims=True deja el resultado como una columna de forma (N, 1) en vez
    #   de una lista (N,), para que se pueda restar fila a fila sin liarse.
    Z = Z - Z.max(axis=1, keepdims=True)

    # [PYTHON] np.exp(Z) calcula e elevado a cada casilla, casilla a casilla.
    E = np.exp(Z)

    # Dividimos cada fila entre su propia suma -> ahora cada fila suma 1.
    # [PYTHON] E.sum(axis=1, keepdims=True) = la suma de cada fila, como columna (N, 1).
    return E / E.sum(axis=1, keepdims=True)


# =============================================================================
# entropia_cruzada  (la función de PÉRDIDA o "loss")
# =============================================================================
# [MODELO] La PÉRDIDA es "la nota de error": un único número que dice cuánto
#   se ha equivocado la red. 0 = perfecto; cuanto mayor, peor. Entrenar es
#   ir reduciendo este número.
#
#   ENTROPÍA CRUZADA = -log( probabilidad que la red dio al género CORRECTO )
#
#       prob. al género correcto | pérdida
#       -------------------------+--------
#              90 %              |  0.105   (lo tenía claro: casi sin error)
#              50 %              |  0.693
#              10 %              |  2.303   (casi como adivinar entre 10)
#               1 %              |  4.605   (muy seguro del género EQUIVOCADO)
#
#   Los fallos muy seguros se castigan muchísimo; los aciertos seguros, casi nada.
#
# [TEMARIO] En regresión lineal se usaba el error cuadrático. Para ELEGIR UNA
#   CATEGORÍA (clasificación) se usa la entropía cruzada, que sale de la
#   perspectiva probabilística (máxima verosimilitud). Con softmax además da
#   un gradiente muy sencillo (ver más abajo).
def entropia_cruzada(Z, y):
    """Calcula la pérdida media y la culpa inicial (dZ) para empezar el backward.

    Z : tabla (N, C) de PUNTUACIONES de la red (todavía sin softmax).
    y : lista (N,) con el género CORRECTO de cada canción, como número 0..C-1.
    Devuelve (perdida, dZ):
        perdida : un número (la media sobre las N canciones).
        dZ      : tabla (N, C), la culpa de cada puntuación.
    """
    # [PYTHON] Z.shape es la forma de la tabla: (N, C). Z.shape[0] es N.
    N = Z.shape[0]

    # El mismo truco de estabilidad de softmax: restar el mayor de cada fila.
    Zs = Z - Z.max(axis=1, keepdims=True)

    # --- La pérdida, calculada de forma estable ---
    # [MODELO] -log(softmax del correcto) = log(suma de e^z) - z_correcto.
    #   (Se obtiene operando con la fórmula de softmax; es exacto y no se
    #   desborda ni da log(0).)
    #
    # [PYTHON] Desglose de la línea:
    #   np.exp(Zs).sum(axis=1)   -> la suma de e^z de cada fila.
    #   np.log(...)              -> el logaritmo natural de cada una.
    #   Zs[np.arange(N), y]      -> para cada canción i, la puntuación de SU
    #                               género correcto y[i]. np.arange(N) es
    #                               [0, 1, ..., N-1] (los números de fila) e y
    #                               son los números de columna: coge una casilla
    #                               por fila.
    #   .mean()                  -> la media de las N canciones.
    perdida = (np.log(np.exp(Zs).sum(axis=1)) - Zs[np.arange(N), y]).mean()

    # --- La culpa inicial dZ ---
    # [TEMARIO] Aplicando la regla de la cadena a  L = -log(p_correcto)  con
    #   p = softmax(z), todas las derivadas se simplifican en algo precioso:
    #           dL/dz_j = p_j - Y_j
    #   donde Y_j vale 1 en el género correcto y 0 en los demás.
    #   Es decir: "la probabilidad que diste menos la que debías dar".
    #     - En el género correcto: p - 1 -> NEGATIVO ("sube esta puntuación").
    #     - En los demás: p - 0 -> POSITIVO ("baja esta puntuación").
    P = softmax(Z)

    # [PYTHON] .copy() crea una copia para no modificar P por error.
    dZ = P.copy()

    # [PYTHON] dZ[np.arange(N), y] -= 1  resta 1 SOLO en la casilla del género
    #   correcto de cada canción. Equivale a calcular P - Y sin construir la
    #   tabla Y entera (que sería un montón de ceros con un 1 por fila).
    dZ[np.arange(N), y] -= 1

    # [MODELO] La pérdida es una MEDIA de N canciones, así que la culpa de cada
    #   canción cuenta 1/N. Si te olvidas de dividir entre N, tus gradientes
    #   salen N veces demasiado grandes (y el gradient check lo detecta).
    # [PYTHON] "/=" divide la tabla entre N "en el sitio".
    dZ /= N

    # [PYTHON] Una función puede devolver VARIAS cosas a la vez separadas por
    #   comas (en realidad devuelve una pareja). Al llamarla las recoges así:
    #       perdida, dZ = entropia_cruzada(Z, y)
    return perdida, dZ
