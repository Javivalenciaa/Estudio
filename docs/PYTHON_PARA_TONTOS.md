# 🐍 Python para tontos (solo lo que usa este repo)

Cada concepto: **qué es → ejemplo → dónde sale en el código.**

## 1. Variables
Una variable es una **caja con nombre** que guarda un valor.
```python
n = 40          # n es 40
nombre = "rock" # texto (siempre entre comillas)
```
`=` significa **"guarda"**, no "es igual a". Para comparar se usa `==`.

## 2. Listas, tuplas y diccionarios
```python
lista  = [64, 32]        # corchetes; se puede modificar
pareja = (W, dW)         # paréntesis: "tupla", se usa para agrupar
dic    = {"mu": 3, "sigma": 5}   # llaves: pares nombre -> valor
lista[0]   # 64  (¡se empieza a contar en 0!)
dic["mu"]  # 3
lista.append(10)  # añade al final
```

## 3. Funciones (`def`)
Una **receta con nombre**.
```python
def sumar(a, b):
    return a + b      # "return" = lo que devuelve
sumar(2, 3)           # 5
```
- Los **espacios al inicio** (sangría, 4 espacios) son OBLIGATORIOS: marcan qué líneas pertenecen a la función.
- Valores por defecto: `def f(x, eps=1e-5)`: si no pasas `eps`, vale `1e-5`.
- Puede devolver varias cosas: `return perdida, dZ` → `perdida, dZ = f(...)`.
- `**extra` recoge argumentos con nombre extra en un diccionario.

## 4. Clases (`class`), `__init__` y `self`
Una clase es un **molde**; un objeto es una **cosa fabricada con el molde**.
```python
class Perro:
    def __init__(self, nombre):   # se ejecuta al crear el perro
        self.nombre = nombre      # guarda el dato DENTRO del objeto
    def ladrar(self):             # un "método": función de la clase
        return self.nombre + " dice guau"

p = Perro("Toby")     # crea un objeto (llama a __init__ solo)
p.ladrar()            # "Toby dice guau"
```
- **`self`** = "este objeto concreto". Siempre es el primer argumento de los métodos; tú no lo escribes al llamar.
- **`self.W = ...`** guarda un dato dentro del objeto → el objeto lo **recuerda** entre llamadas (por eso `forward` guarda `self.X` para que `backward` lo use).
- Una variable sin `self.` muere al terminar la función.
- Cada objeto tiene sus propios datos: dos `Lineal(...)` tienen pesos distintos.

En el repo: `Lineal`, `ReLU`, `Red`.

## 5. Importar código
```python
import numpy as np            # trae NumPy y lo apoda np
from .capas import Lineal     # trae SOLO la clase Lineal del archivo capas.py de esta carpeta
```
El `.` delante significa "esta misma carpeta". El archivo `__init__.py` vacío convierte una carpeta en "paquete" importable.

## 6. Bucles y comprensiones
```python
for capa in self.capas:       # repite para cada elemento
    X = capa.forward(X)

for i, (p, _) in enumerate(lista):   # enumerate da (0, x0), (1, x1)...
    ...                              # "_" = valor que ignoro

reversed(lista)               # recorre al revés
range(10)                     # 0, 1, ..., 9

# lista por comprensión (bucle en una línea):
[2 * x for x in [1, 2, 3]]    # [2, 4, 6]
```

## 7. NumPy: tablas de números
NumPy trabaja con **arrays** (tablas) de cualquier dimensión.
```python
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])
a.shape        # (2, 3)  -> 2 filas, 3 columnas
a.T            # traspuesta: gira filas y columnas -> forma (3, 2)
np.zeros(4)            # [0. 0. 0. 0.]
np.zeros_like(a)       # ceros con la forma de a
np.arange(3)           # [0 1 2]
```

### Operaciones que usa el repo
| Código | Qué hace |
|---|---|
| `A @ B` | **producto de matrices** (columnas de A = filas de B) |
| `A * B` | multiplicación **casilla a casilla** (no es `@`) |
| `X > 0` | tabla de True/False |
| `X * (X > 0)` | True cuenta como 1 y False como 0 |
| `A.sum(axis=0)` | suma **hacia abajo** (una suma por columna) |
| `A.sum(axis=1)` | suma **hacia la derecha** (una suma por fila) |
| `A.max(axis=1, keepdims=True)` | máximo de cada fila, como columna `(N,1)` |
| `A.argmax(axis=1)` | **posición** del máximo de cada fila |
| `np.exp(A)`, `np.log(A)`, `np.sqrt(A)` | operaciones casilla a casilla |
| `A.mean()` | media de todas las casillas |
| `A.copy()` | copia independiente |
| `A[np.arange(N), y]` | casilla `y[i]` de cada fila `i` |
| `A[np.arange(N), y] -= 1` | resta 1 solo en esas casillas |
| `A[...] = B` | rellena A **en el sitio** con B |

### Broadcasting
Si sumas una tabla `(N, 64)` con un vector `(64,)`, NumPy **repite** el vector en cada fila. Por eso `X @ W + b` funciona.

### Formas: tu mejor detector de errores
`(N, 40) @ (40, 64) = (N, 64)`. Si no cuadra, NumPy da error. Cuando no sepas si poner `.T`, **comprueba las formas**: el gradiente de algo tiene la forma de ese algo.

## 8. `+=`, `-=`, `/=`
`x += 1` es `x = x + 1`. Con arrays, `p -= lr * g` **modifica el array original** (lo que necesitamos para cambiar los botones de la red). `p = p - lr * g` crearía otro array y la red no se enteraría.

## 9. Copias vs. mismo objeto
```python
a = np.array([1, 2]); b = a        # b y a son EL MISMO objeto
b[0] = 99                          # a también cambia
c = a.copy()                       # c es independiente
```
Importante en `params()` (devuelve los objetos reales) y en `gradcheck` (copia las pendientes).

## 10. Notación científica
`1e-5` = 0.00001; `2e3` = 2000.

## 11. Ejecutar un archivo
```
python ejemplos/01_probar_capas.py
```
desde la carpeta principal del repo. Si ves `ModuleNotFoundError: No module named 'nn'`, estás en la carpeta equivocada.

## 12. Errores típicos
| Error | Causa |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'zeros_line'` | Errata: es `zeros_like` |
| `shapes (5,40) and (64,40) not aligned` | Dimensiones mal: revisa `@` y `.T` |
| `IndentationError` | Sangría mal (usa 4 espacios) |
| `NameError: name 'x' is not defined` | Variable sin crear o mal escrita |
| `ModuleNotFoundError` | Falta instalar (`pip install numpy`) o ejecutas desde la carpeta equivocada |
