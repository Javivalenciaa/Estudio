# =============================================================================
# ejemplos/datos_falsos.py -> canciones INVENTADAS para probar tu código
# =============================================================================
# [MODELO] Tú (Persona A) no tienes todavía los datos reales (los prepara la
#   Persona C con GTZAN). Para probar la red generamos canciones falsas: cada
#   "género" es un grupito de puntos alrededor de un centro al azar.
#   Si tu red es correcta, debe poder separar estos grupitos.

import numpy as np


def datos_falsos(n_por_clase=30, n_rasgos=40, n_clases=10, semilla=0):
    """Devuelve X (N, n_rasgos) e y (N,) con N = n_por_clase * n_clases."""
    rng = np.random.default_rng(semilla)

    # Un "centro" al azar por cada género: tabla (n_clases, n_rasgos).
    centros = rng.standard_normal((n_clases, n_rasgos)) * 2.0

    X, y = [], []
    for c in range(n_clases):  # [PYTHON] range(10) da 0,1,...,9
        # Canciones del género c = su centro + un poco de ruido.
        puntos = centros[c] + rng.standard_normal((n_por_clase, n_rasgos))
        X.append(puntos)
        y += [c] * n_por_clase  # [PYTHON] [3] * 4 = [3, 3, 3, 3]

    # [PYTHON] np.vstack apila tablas una encima de otra en una sola.
    return np.vstack(X), np.array(y)
