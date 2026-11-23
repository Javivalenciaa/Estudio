# =============================================================================
# nn/capas.py  ->  LAS CAPAS DE LA RED (clases Lineal y ReLU)
# =============================================================================
#
# QUÉ HAY EN ESTE ARCHIVO
#   Dos "piezas" que se encadenan para formar la red neuronal:
#     1) Lineal : hace la cuenta   Z = X @ W + b   (la que tiene botones que se aprenden)
#     2) ReLU   : "dobla" los números (deja pasar lo positivo y pone a 0 lo negativo)
#
# CÓMO LEER ESTE ARCHIVO
#   - Las líneas que empiezan por "#" son COMENTARIOS: Python las ignora por
#     completo. Están para que las leas tú.
#   - Cuando ves  [PYTHON]  te explico algo del lenguaje Python.
#   - Cuando ves  [MODELO]  te explico qué hace eso dentro de la red neuronal.
#   - Cuando ves  [TEMARIO] te lo conecto con lo que se da en clase.
#
# Si no has visto nunca Python, lee antes docs/PYTHON_PARA_TONTOS.md.
# =============================================================================


# [PYTHON] "import" trae código que ya escribió otra gente para poder usarlo.
#   "numpy" es una librería para hacer cuentas con tablas de números muy rápido.
#   "as np" le pone un apodo corto: a partir de aquí escribimos "np" en vez de "numpy".
import numpy as np


# =============================================================================
# CLASE 1: Lineal
# =============================================================================
# [PYTHON] Una "class" (clase) es un MOLDE para fabricar objetos.
#   Piensa en un molde de galletas: la clase es el molde y cada galleta es un
#   "objeto". Todas las galletas tienen la misma forma, pero cada una puede
#   tener sus propios datos (una con chocolate, otra quemada...).
#
#   Aquí el molde es "Lineal". Cuando escribamos  Lineal(40, 64)  fabricaremos
#   UNA capa concreta. Si escribimos Lineal(...) tres veces, tendremos tres
#   capas distintas, cada una con SUS propios botones. Eso es lo bueno de las
#   clases: cada objeto RECUERDA sus propios datos.
#
# [MODELO] Una capa lineal es el paso más básico de una red neuronal:
#     Z = X @ W + b
#   donde:
#     X = tabla de entrada. Una fila por canción, una columna por rasgo.
#     W = tabla de "pesos" (los botones principales). Se aprenden entrenando.
#     b = "sesgo": un número extra por salida que se suma. También se aprende.
#     Z = la salida (una puntuación por cada número de salida).
#
# [TEMARIO] Es exactamente la regresión lineal  Xw + b  de los apuntes, solo que
#   en vez de un vector w (una sola salida) usamos una MATRIZ W (varias salidas
#   a la vez). Una red neuronal es, en el fondo, regresiones lineales apiladas
#   con "dobleces" (ReLU) entre medias.
class Lineal:
    """Capa lineal: Z = X @ W + b.

    [PYTHON] Este texto entre tres comillas justo debajo del nombre de la clase
    se llama "docstring". Es la descripción oficial de la clase. No hace nada
    al ejecutar; si escribes help(Lineal) en Python, lo muestra.
    """

    # -------------------------------------------------------------------------
    # __init__ : se ejecuta UNA vez, automáticamente, al crear la capa
    # -------------------------------------------------------------------------
    # [PYTHON] "def" define una FUNCIÓN: una receta con nombre. Cuando una
    #   función está dentro de una clase se llama MÉTODO.
    #   "__init__" (dos guiones bajos, init, dos guiones bajos) es un método
    #   ESPECIAL: Python lo ejecuta SOLO en cuanto haces  Lineal(...).
    #   Sirve para dejar el objeto recién creado en su estado inicial.
    #
    # [PYTHON] "self" significa "ESTE objeto concreto, el que se está creando
    #   o el que está trabajando ahora". Es siempre el primer argumento de los
    #   métodos. Tú no lo escribes al llamar: Python lo pone solo.
    #   Si haces  capa1 = Lineal(40, 64),  dentro de __init__ "self" ES capa1.
    #
    # Argumentos que recibe:
    #   n_in  : cuántos números ENTRAN por canción (ej. 40 rasgos).
    #   n_out : cuántos números SALEN por canción (ej. 64 neuronas).
    #   rng   : un generador de números al azar. "=None" significa que es
    #           OPCIONAL: si no lo pasas, vale None (nada).
    def __init__(self, n_in, n_out, rng=None):
        # [PYTHON] "or" devuelve el primero de los dos que "valga algo".
        #   Si rng vale None (no nos lo han dado), usa el de la derecha:
        #   crea un generador nuevo con semilla 0.
        # [MODELO] La SEMILLA (0) hace que el "azar" sea siempre el mismo.
        #   Así, si ejecutas el programa dos veces, los números "aleatorios"
        #   salen idénticos y los resultados se pueden repetir y comparar.
        rng = rng or np.random.default_rng(0)

        # --- Los pesos W ---------------------------------------------------
        # [MODELO] W es la tabla de botones. Forma (n_in, n_out): una fila por
        #   cada número de entrada y una columna por cada número de salida.
        #   Ej: Lineal(40, 64) -> W tiene 40 filas y 64 columnas (2560 botones).
        #   Al principio se rellenan con números AL AZAR: la red no sabe nada
        #   y entrenar consiste en ir ajustándolos.
        #
        # [PYTHON] rng.standard_normal((a, b)) crea una tabla de a filas y b
        #   columnas con números al azar siguiendo una "campana de Gauss"
        #   (casi todos cerca de 0, pocos muy grandes, positivos y negativos).
        #   Fíjate en el doble paréntesis: ((n_in, n_out)) -> el interior es
        #   UNA pareja (tupla) que le dice la forma de la tabla.
        #
        # [MODELO] "* np.sqrt(2.0 / n_in)" encoge esos números cuantas más
        #   entradas haya. Es la inicialización de He. Si los pesos fueran muy
        #   grandes, los números crecerían sin control al pasar por varias
        #   capas; si fueran muy pequeños, se apagarían hasta 0. Esta
        #   escala mantiene los números en un tamaño razonable con ReLU.
        #   np.sqrt = raíz cuadrada.
        #
        # [PYTHON] "self.W = ..." guarda la tabla DENTRO del objeto. A partir de
        #   ahora la capa "recuerda" su W mientras exista. (Una variable
        #   normal, sin "self.", desaparecería al acabar la función.)
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)

        # --- Los sesgos b --------------------------------------------------
        # [MODELO] Un número por salida, todos a 0 al principio. Es un
        #   "valor de partida" que se suma a cada puntuación.
        # [PYTHON] np.zeros(n) crea una lista (array) de n ceros. Forma (n_out,).
        self.b = np.zeros(n_out)

        # --- Sitio para las pendientes ------------------------------------
        # [MODELO] dW y db son las PENDIENTES (gradientes) de W y b: dicen,
        #   para cada botón, hacia dónde hay que girarlo para reducir el error.
        #   La "d" significa "cuánto cambia el error si muevo esto un poquito".
        #   Aquí solo reservamos el sitio (todo ceros); se rellenan en backward.
        # [PYTHON] np.zeros_like(t) = "una tabla de ceros con la MISMA forma
        #   que la tabla t". Ojo con el nombre: es zeros_LIKE (como), no "line".
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    # -------------------------------------------------------------------------
    # forward : HACIA DELANTE. Calcula la salida de la capa.
    # -------------------------------------------------------------------------
    # [MODELO] "Forward" = pasar los datos por la capa para obtener la respuesta.
    #   Recibe X, de forma (N, n_in): N canciones, n_in rasgos cada una.
    #   Devuelve Z, de forma (N, n_out).
    def forward(self, X):
        # [MODELO] Guardamos la entrada. ¡Esta línea es clave! El backward
        #   (más abajo) necesita saber qué entró para calcular las pendientes.
        # [PYTHON] Si no la guardáramos en "self.X", al acabar forward la
        #   variable X desaparecería y backward no podría usarla. Por eso esto
        #   es una clase y no una simple función: las clases RECUERDAN.
        self.X = X

        # [MODELO] La cuenta de la capa: Z = X @ W + b.
        # [PYTHON] "@" es el PRODUCTO DE MATRICES: multiplica cada fila de X por
        #   cada columna de W y suma. Para que se pueda, las columnas de X
        #   (n_in) deben ser tantas como las filas de W (n_in). El resultado
        #   tiene (filas de X, columnas de W) = (N, n_out).
        #   Forma:  (N, n_in) @ (n_in, n_out) = (N, n_out)
        #   El "+ self.b" suma el sesgo a TODAS las filas (NumPy repite b en
        #   cada fila automáticamente; se llama "broadcasting").
        # [PYTHON] "return" = "esto es lo que devuelve la función".
        return X @ self.W + self.b

    # -------------------------------------------------------------------------
    # backward : HACIA ATRÁS. Reparte la culpa del error entre los botones.
    # -------------------------------------------------------------------------
    # [MODELO] Cuando la red se equivoca, hay que saber CUÁNTA CULPA tiene cada
    #   botón para girarlo en la dirección correcta. Eso es "backpropagation".
    #   Esta capa RECIBE dZ = la culpa de cada número que ella produjo (viene
    #   de la capa siguiente, o de la función de pérdida si es la última).
    #   Forma de dZ: (N, n_out), igual que Z.
    #
    # [TEMARIO] Es la REGLA DE LA CADENA de las derivadas: si el error depende
    #   de Z y Z depende de W, entonces dError/dW = (dError/dZ) * (dZ/dW).
    #   Con Z = X@W + b, esa regla da justo las tres fórmulas de abajo.
    def backward(self, dZ):
        # --- Pendiente de los pesos ---
        # [MODELO] Un peso W[k, j] conecta la entrada k con la salida j. Es más
        #   culpable cuanto mayor era la entrada k y cuanto más culpa tiene la
        #   salida j. Sumando sobre todas las canciones sale:  dW = X.T @ dZ
        # [PYTHON] ".T" es la TRASPUESTA: gira la tabla (filas <-> columnas).
        #   Si X es (N, n_in), X.T es (n_in, N).
        #   Formas:  (n_in, N) @ (N, n_out) = (n_in, n_out)  <- ¡la forma de W!
        #   TRUCO: un resultado de gradiente SIEMPRE tiene la forma del botón
        #   al que corresponde. Si dudas con los ".T", mira las formas.
        self.dW = self.X.T @ dZ

        # --- Pendiente de los sesgos ---
        # [MODELO] El sesgo se sumó igual a todas las canciones, así que su
        #   culpa es la suma de la culpa de todas las canciones.
        # [PYTHON] .sum(axis=0) suma "hacia abajo": colapsa las filas y deja una
        #   suma por cada columna. Forma resultante: (n_out,).
        self.db = dZ.sum(axis=0)

        # --- Culpa que se pasa a la capa de ATRÁS ---
        # [MODELO] La entrada X de esta capa era la salida de la capa anterior,
        #   así que ella también tiene culpa y necesita saberla.
        #   dX = dZ @ W.T  con formas  (N, n_out) @ (n_out, n_in) = (N, n_in).
        #   Fíjate: es de forma (N, n_in), igual que X. Eso es lo que hace
        #   que la cadena de capas encaje de atrás hacia delante.
        return dZ @ self.W.T

    # -------------------------------------------------------------------------
    # params : entrega los botones y sus pendientes al optimizador
    # -------------------------------------------------------------------------
    # [MODELO] El OPTIMIZADOR (lo escribe la Persona B) es quien gira los
    #   botones. Para eso necesita, de cada botón, el propio botón y su
    #   pendiente. Este método se los da.
    # [PYTHON] Devuelve una LISTA (entre corchetes) de PAREJAS (entre paréntesis,
    #   se llaman tuplas): [(botón1, pendiente1), (botón2, pendiente2)].
    #   Importante: devuelve las tablas MISMAS (no copias), así que cuando el
    #   optimizador las modifique, cambiará de verdad los botones de la capa.
    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


# =============================================================================
# CLASE 2: ReLU
# =============================================================================
# [MODELO] ReLU significa "Rectified Linear Unit". Es una función de
#   ACTIVACIÓN: se pone entre dos capas lineales. Hace esto con cada número:
#         si x > 0  ->  se queda igual
#         si x <= 0 ->  pasa a ser 0
#
#   ¿Para qué? Si apilas capas que solo hacen X@W+b, el resultado total sigue
#   siendo "una sola capa lineal" (multiplicar matrices una tras otra es otra
#   multiplicación de matrices). Entonces la red solo podría aprender fronteras
#   RECTAS. ReLU "dobla" la línea y permite aprender formas más complicadas.
#
#   Una "neurona" está ENCENDIDA si su salida es > 0 y APAGADA si vale 0.
class ReLU:
    """Activación: deja pasar lo positivo y pone a 0 lo negativo."""

    # Esta clase NO tiene __init__ porque no tiene botones que preparar.
    # [PYTHON] Si una clase no necesita estado inicial, no hace falta escribirlo.

    def forward(self, X):
        # [PYTHON] "X > 0" compara cada casilla con 0 y devuelve una tabla del
        #   mismo tamaño llena de True (verdadero) y False (falso).
        # [MODELO] Esa tabla de verdadero/falso es la "máscara" de neuronas
        #   encendidas. La guardamos porque backward la va a necesitar.
        self.mascara = X > 0

        # [PYTHON] Multiplicar un número por True es multiplicarlo por 1 (se
        #   queda igual) y por False es multiplicarlo por 0 (pasa a 0).
        #   Eso es exactamente ReLU. (A veces sale "-0.", un cero con signo:
        #   vale lo mismo que 0, no es un error.)
        return X * self.mascara

    def backward(self, dA):
        # [MODELO] La culpa solo pasa por las neuronas que estaban ENCENDIDAS.
        #   Si una neurona estaba apagada (salida 0), mover sus botones no
        #   habría cambiado nada, así que no tiene culpa: su dA se convierte
        #   en 0. Multiplicar por la máscara lo hace de golpe.
        # [TEMARIO] Es la derivada de ReLU: vale 1 donde x>0 y 0 donde x<=0.
        return dA * self.mascara

    def params(self):
        # ReLU no tiene botones, así que devuelve una lista VACÍA. Existe para
        # que la Red pueda preguntar "¿cuáles son tus botones?" a cualquier
        # capa sin tener que distinguir si es Lineal o ReLU.
        return []
