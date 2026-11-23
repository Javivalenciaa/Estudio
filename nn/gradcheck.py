# nn/gradcheck.py: comprueba que los backward calculan bien los gradientes
# comparándolos con la derivada numérica (diferencias finitas).

import numpy as np
from .perdidas import entropia_cruzada


def comprobar_gradientes(red, X, y, eps=1e-5):
    """Devuelve el peor error relativo entre backprop y gradiente numérico.
    Debe salir por debajo de 1e-6."""
    _, dZ = entropia_cruzada(red.forward(X), y)
    red.backward(dZ)
    pares = [(p, g.copy()) for p, g in red.params()]

    peor = 0.0
    for p, g_backprop in pares:
        g_num = np.zeros_like(p)
        for idx in np.ndindex(*p.shape):
            viejo = p[idx]
            p[idx] = viejo + eps
            L_mas, _ = entropia_cruzada(red.forward(X), y)
            p[idx] = viejo - eps
            L_menos, _ = entropia_cruzada(red.forward(X), y)
            p[idx] = viejo
            g_num[idx] = (L_mas - L_menos) / (2 * eps)

        num = np.abs(g_backprop - g_num).max()
        den = np.abs(g_backprop).max() + np.abs(g_num).max() + 1e-12
        peor = max(peor, num / den)

    return peor
