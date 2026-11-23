# =============================================================================
# nn/gradcheck.py  ->  EL DETECTOR DE ERRORES (gradient check)
# =============================================================================
#
# EL PROBLEMA QUE RESUELVE
#   Si te equivocas en un ".T" o te olvidas de dividir entre N en un backward,
#   la red SEGUIRÁ EJECUTÁNDOSE sin dar ningún error... pero aprenderá mal y no
#   sabrás por qué. Aquí hay una forma de comprobar que los gradientes que
#   calculan tus backward son correctos.
#
# LA IDEA (en cinco pasos)
#   Para un botón cualquiera (por ejemplo W[0, 0]):
#     1. Súmale un poquito (eps = 0.00001) y calcula la pérdida.
#     2. Réstale ese poquito y calcula la pérdida otra vez.
#     3. La pendiente "de verdad" es
#           (pérdida_arriba - pérdida_abajo) / (2 * eps)
#        Es lenta pero segura: NO usa tus backward, solo el forward.
#     4. Compárala con la pendiente que calculó tu backward.
#     5. Repite con TODOS los botones.
#   Si tu código está bien, las dos pendientes son casi idénticas.
#
# [TEMARIO] Es la definición de DERIVADA como límite:
#       f'(x) ≈ ( f(x + h) - f(x - h) ) / (2h)    para h muy pequeño.
# =============================================================================

import numpy as np
from .perdidas import entropia_cruzada


def comprobar_gradientes(red, X, y, eps=1e-5):
    """Compara el gradiente de backprop con uno numérico (diferencias finitas).

    Devuelve el PEOR error relativo entre los dos. Debe salir por debajo de
    1e-6 (un millonésimo). Si sale 1e-3 o más, hay un error en algún backward.

    [PYTHON] "1e-5" es notación científica: 1 por 10 elevado a -5 = 0.00001.
    """
    # --- 1) Gradientes "oficiales" de nuestro backprop ---
    # Pasada normal: forward -> pérdida y dZ -> backward. Las pendientes de cada
    # botón quedan guardadas dentro de las capas.
    # [PYTHON] "_, dZ = ..." el "_" recoge un valor que no vamos a usar (la pérdida).
    _, dZ = entropia_cruzada(red.forward(X), y)
    red.backward(dZ)

    # Guardamos una COPIA de las pendientes de backprop, porque al repetir
    # forward/backward más abajo se podrían sobrescribir.
    # [PYTHON] g.copy() duplica la tabla. "pares" es una lista de parejas
    #   (botón, copia de su pendiente de backprop).
    pares = [(p, g.copy()) for p, g in red.params()]

    # "peor" guardará el mayor error encontrado en todos los botones.
    peor = 0.0

    for p, g_backprop in pares:
        # Tabla donde iremos apuntando la pendiente NUMÉRICA de cada casilla
        # del botón p.
        g_num = np.zeros_like(p)

        # [PYTHON] np.ndindex(*p.shape) recorre TODAS las posiciones de una
        #   tabla. Si p es de forma (40, 64) da (0,0), (0,1), ..., (39,63).
        #   Si p es de forma (64,) da (0,), (1,), ... "*p.shape" desempaqueta
        #   la forma en argumentos separados.
        for idx in np.ndindex(*p.shape):
            viejo = p[idx]  # guardamos el valor original de esta casilla

            # Paso 1: subir un poquito y medir la pérdida.
            # [PYTHON] Modificamos p "en el sitio": como p es el MISMO objeto que
            #   usa la capa, red.forward ya ve el cambio.
            p[idx] = viejo + eps
            L_mas, _ = entropia_cruzada(red.forward(X), y)

            # Paso 2: bajar un poquito y medir la pérdida.
            p[idx] = viejo - eps
            L_menos, _ = entropia_cruzada(red.forward(X), y)

            # ¡IMPORTANTE! Devolver el valor original. Si no, dejarías el
            # botón cambiado y estropearías la red.
            p[idx] = viejo

            # Paso 3: la pendiente numérica de esta casilla.
            g_num[idx] = (L_mas - L_menos) / (2 * eps)

        # --- Comparar las dos pendientes de este botón ---
        # Error "absoluto": la mayor diferencia entre las dos tablas.
        num = np.abs(g_backprop - g_num).max()
        # Se divide entre el tamaño de las pendientes para obtener un error
        # RELATIVO (si las pendientes son enormes, una diferencia pequeña no
        # importa). El "+ 1e-12" evita dividir entre 0 si las dos son nulas.
        den = np.abs(g_backprop).max() + np.abs(g_num).max() + 1e-12

        # Nos quedamos con el peor error visto hasta ahora.
        peor = max(peor, num / den)

    return peor
