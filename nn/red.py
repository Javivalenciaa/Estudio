# =============================================================================
# nn/red.py  ->  LA RED COMPLETA (cadena de capas) + guardar/cargar
# =============================================================================
#
# QUÉ HAY EN ESTE ARCHIVO
#   - class Red       : una LISTA de capas que se ejecutan una tras otra.
#   - crear_mlp(...)  : una "fábrica" que monta la lista de capas por ti.
#   - guardar / cargar_pesos : guardar en un archivo los botones ya entrenados.
#
# [MODELO] MLP = "Multi-Layer Perceptron" = red de varias capas. En nuestro
#   proyecto:   40 rasgos -> [Lineal 64] -> ReLU -> [Lineal 32] -> ReLU -> [Lineal 10]
#   La última capa devuelve 10 puntuaciones, una por género.
# =============================================================================

import numpy as np

# [PYTHON] "from .capas import Lineal, ReLU" trae esas dos clases de OTRO archivo.
#   El PUNTO antes de "capas" significa "del archivo capas.py que está en esta
#   misma carpeta (nn)". Así se reparte el código en varios archivos.
from .capas import Lineal, ReLU
from .perdidas import softmax


# =============================================================================
# class Red
# =============================================================================
class Red:
    """Una lista de capas que se aplican una detrás de otra."""

    def __init__(self, capas):
        # [PYTHON] "capas" es una LISTA de objetos, por ejemplo
        #   [Lineal(...), ReLU(), Lineal(...), ReLU(), Lineal(...)].
        #   La guardamos en self para que la red la recuerde.
        self.capas = capas

    # -------------------------------------------------------------------------
    # forward: pasar las canciones por TODAS las capas, en orden
    # -------------------------------------------------------------------------
    def forward(self, X):
        # [PYTHON] "for ... in ..." es un BUCLE: repite lo de dentro una vez por
        #   cada elemento de la lista. Lo que está SANGRADO (4 espacios a la
        #   derecha) es lo que se repite.
        for capa in self.capas:
            # [MODELO] La salida de una capa es la entrada de la siguiente.
            #   Reutilizamos el nombre X: en cada vuelta X pasa a valer el
            #   resultado de la última capa que se ha aplicado.
            X = capa.forward(X)
        # Al acabar, X son las puntuaciones finales Z, forma (N, n_clases).
        return X

    # -------------------------------------------------------------------------
    # backward: repartir la culpa hacia atrás por TODAS las capas
    # -------------------------------------------------------------------------
    def backward(self, dZ):
        # [PYTHON] reversed(lista) recorre la lista AL REVÉS (de la última a la
        #   primera). Necesario porque la culpa empieza en la salida y viaja
        #   hacia la entrada.
        for capa in reversed(self.capas):
            # [MODELO] Cada capa recibe la culpa de la de delante, calcula la
            #   de sus botones (queda guardada dentro de la capa) y DEVUELVE la
            #   culpa que le toca a la de atrás.
            dZ = capa.backward(dZ)
        return dZ

    # -------------------------------------------------------------------------
    # params: juntar los botones de TODAS las capas en una sola lista
    # -------------------------------------------------------------------------
    def params(self):
        # [PYTHON] Esto es una "list comprehension": una forma compacta de
        #   construir una lista con un bucle. Se lee:
        #     "para cada capa, y para cada pareja que devuelve su params(),
        #      mete esa pareja en la lista".
        #   Equivale a:
        #       todos = []
        #       for capa in self.capas:
        #           for par in capa.params():
        #               todos.append(par)
        #       return todos
        #   (ReLU devuelve una lista vacía, así que no aporta nada.)
        return [par for capa in self.capas for par in capa.params()]

    # -------------------------------------------------------------------------
    # predecir_proba: probabilidades por género
    # -------------------------------------------------------------------------
    def predecir_proba(self, X):
        # forward da PUNTUACIONES; softmax las convierte en PROBABILIDADES.
        return softmax(self.forward(X))

    # -------------------------------------------------------------------------
    # predecir: el género ganador de cada canción
    # -------------------------------------------------------------------------
    def predecir(self, X):
        # [PYTHON] .argmax(axis=1) devuelve, para cada fila, la POSICIÓN del
        #   número mayor. Aquí la posición = el número del género con más
        #   puntuación. (No hace falta softmax: el mayor de las puntuaciones es
        #   también la mayor probabilidad.) Devuelve forma (N,).
        return self.forward(X).argmax(axis=1)


# =============================================================================
# crear_mlp: la "fábrica" de redes
# =============================================================================
# [PYTHON] Esta es una FUNCIÓN normal (no pertenece a ninguna clase). Recibe
#   cómo quieres la red y devuelve un objeto Red ya montado.
#
# Argumentos:
#   n_in    : cuántos números entran (40 rasgos).
#   ocultas : LISTA con el tamaño de cada capa intermedia. [64, 32] = dos capas
#             ocultas, de 64 y 32 neuronas. [] = SIN capas ocultas.
#   n_out   : cuántos números salen (10 géneros).
#   semilla : la semilla del azar, para que sea repetible.
#
# Ejemplos:
#   crear_mlp(40, [64, 32], 10)  ->  40 -> 64 -> 32 -> 10   (nuestra red; 5034 botones)
#   crear_mlp(40, [],       10)  ->  40 -> 10               (modelo lineal = regresión
#                                                             softmax; 410 botones)
def crear_mlp(n_in, ocultas, n_out, semilla=0):
    """crear_mlp(40, [64, 32], 10) -> 40 -> 64 -> 32 -> 10, con ReLU entre capas."""
    # Un único generador de azar compartido por todas las capas: cada capa recibe
    # números distintos pero el conjunto es reproducible.
    rng = np.random.default_rng(semilla)

    # [PYTHON] "capas, previo = [], n_in" es UNA línea que crea dos variables a
    #   la vez: capas = [] (lista vacía) y previo = n_in. "previo" recordará el
    #   tamaño de salida de la última capa añadida (que será la entrada de la
    #   siguiente).
    capas, previo = [], n_in

    # Una capa oculta por cada número de la lista "ocultas".
    for h in ocultas:
        # [PYTHON] "+=" con una lista AÑADE esos elementos al final. Aquí añade
        #   dos cosas: una capa Lineal y un ReLU justo detrás.
        capas += [Lineal(previo, h, rng), ReLU()]
        previo = h  # la siguiente capa recibirá h números

    # La última capa lleva a n_out (10 géneros) y SIN ReLU ni softmax:
    #   - sin ReLU porque queremos puntuaciones libres (pueden ser negativas);
    #   - sin softmax porque ya va dentro de la función de pérdida.
    # [PYTHON] .append(x) añade un elemento al final de la lista.
    capas.append(Lineal(previo, n_out, rng))

    return Red(capas)


# =============================================================================
# guardar / cargar_pesos
# =============================================================================
# [MODELO] Cuando B termina de entrenar, los botones tienen los valores buenos.
#   Hay que guardarlos en un archivo para no reentrenar cada vez y para que la
#   web de la Persona C pueda usar la red.
def guardar(red, ruta, **extra):
    """Guarda pesos y datos extra (media, desviación, nombres de géneros...) en un .npz."""
    # [PYTHON] "**extra" en la definición significa "acepta cualquier número de
    #   argumentos con nombre y guárdalos en un diccionario llamado extra".
    #   Ej: guardar(red, "m.npz", mu=..., sigma=...) -> extra = {"mu":..., "sigma":...}.
    #
    # [PYTHON] Un DICCIONARIO es una colección de pares nombre -> valor, entre
    #   llaves { }. Lo que sigue es una "dict comprehension" (como la list
    #   comprehension pero produce un diccionario):
    #     - red.params() es la lista de parejas (botón, pendiente).
    #     - enumerate(...) numera cada pareja: (0, pareja0), (1, pareja1), ...
    #     - "(p, _)" desempaqueta la pareja: p = el botón; "_" = la pendiente,
    #       que no nos interesa (el guion bajo significa "no lo uso").
    #     - f"p{i}" es un TEXTO con el número dentro: "p0", "p1", "p2"...
    datos = {f"p{i}": p for i, (p, _) in enumerate(red.params())}

    # .update(otro) añade al diccionario todo lo del otro diccionario.
    datos.update(extra)

    # [PYTHON] np.savez guarda varias tablas con nombre en un solo archivo .npz.
    #   "**datos" "desenvuelve" el diccionario en argumentos con nombre.
    np.savez(ruta, **datos)


def cargar_pesos(red, ruta):
    """Lee un .npz guardado y mete los botones dentro de una red YA CREADA."""
    # allow_pickle=False es una medida de seguridad: no permite cargar objetos
    # arbitrarios de Python desde el archivo, solo tablas de números y textos.
    d = np.load(ruta, allow_pickle=False)

    for i, (p, _) in enumerate(red.params()):
        # [PYTHON] "p[...] = valor" RELLENA la tabla p que ya existe con los
        #   valores nuevos, en el mismo sitio de memoria. Si escribieras
        #   "p = valor" crearías una tabla nueva y la capa seguiría con los
        #   números viejos sin enterarse. ("..." se llama Ellipsis y aquí
        #   significa "todas las casillas".)
        p[...] = d[f"p{i}"]

    # Devolvemos el contenido por si hace falta leer los datos extra
    # (media, desviación, nombres de géneros...).
    return d
