# =============================================================================
# ejemplos/02_gradcheck_y_memorizar.py -> ¿está BIEN tu red?
# =============================================================================
# Ejecutar:   python ejemplos/02_gradcheck_y_memorizar.py
#
# Hace dos pruebas que debe pasar TU parte (Persona A) antes de dársela a B:
#   PRUEBA 1: gradient check (los gradientes coinciden con los numéricos).
#   PRUEBA 2: la red es capaz de MEMORIZAR 20 canciones con etiquetas al azar.
#             Si no puede memorizar, algo está roto; si puede, la red tiene
#             capacidad y el backward empuja en la dirección correcta.

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nn.red import crear_mlp, guardar, cargar_pesos
from nn.perdidas import entropia_cruzada
from nn.gradcheck import comprobar_gradientes
from ejemplos.datos_falsos import datos_falsos

# ---------------------------------------------------------------------------
print("=== PRUEBA 1: gradient check ===")
X, y = datos_falsos(n_por_clase=3, n_rasgos=40, n_clases=10)
red = crear_mlp(40, [64, 32], 10)
err = comprobar_gradientes(red, X, y)
print("error relativo máximo:", err)
print("RESULTADO:", "OK ✅" if err < 1e-6 else "MAL ❌ (revisa tus backward)")
# DEBE SALIR un número del orden de 1e-10 y OK.

# ---------------------------------------------------------------------------
print("\n=== PRUEBA 2: memorizar 20 canciones con etiquetas al azar ===")
rng = np.random.default_rng(0)
Xm = rng.standard_normal((20, 40))
ym = rng.integers(0, 10, size=20)  # etiquetas al azar entre 0 y 9
red = crear_mlp(40, [64, 32], 10)

# Mini-entrenamiento a mano con descenso de gradiente (lo que luego hará la
# Persona B de forma más elegante con su optimizador):
#   para cada botón p con pendiente g:  p = p - lr * g
lr = 0.1
for it in range(500):
    perdida, dZ = entropia_cruzada(red.forward(Xm), ym)
    red.backward(dZ)
    for p, g in red.params():
        # [PYTHON] "-=" resta EN EL SITIO. Es imprescindible para que cambie el
        #   botón de verdad (p = p - ... crearía una variable nueva).
        p -= lr * g

acc = (red.predecir(Xm) == ym).mean()
print("pérdida final:", round(perdida, 4), " precisión:", acc)
print("RESULTADO:", "OK ✅" if acc == 1.0 else "MAL ❌")
# DEBE SALIR: precisión 1.0 y OK.

# ---------------------------------------------------------------------------
print("\n=== PRUEBA 3: guardar y cargar ===")
guardar(red, "/tmp/prueba_modelo.npz")
red2 = crear_mlp(40, [64, 32], 10, semilla=99)  # otra red con otros botones
cargar_pesos(red2, "/tmp/prueba_modelo.npz")
igual = np.allclose(red.forward(Xm), red2.forward(Xm))
print("la red cargada da lo mismo que la guardada:", igual)
print("RESULTADO:", "OK ✅" if igual else "MAL ❌")
