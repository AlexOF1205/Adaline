import numpy as np
# Parámetros iniciales
n = 0.1             
w1 = np.random.rand()  
w2 = np.random.rand()
b = np.random.rand()
max_epocas = 20    


print("Selecciona la función de pérdida a utilizar:")
print("1 - (y - d)")
print("2 - (y - d)^2")
print("3 - 1/4 * (y - d)^2")

opcion = input("Ingresa el número de la opción (1/2/3): ")

# Tabla (entrada, salida deseada)
datos = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

print("\nENTRENAMIENTO ADALINE")
print(f"Usando función de pérdida tipo {opcion}\n")

for epoca in range(1, max_epocas + 1):
    print(f"\n=== Época {epoca} ===")
    for paso, (x1, x2, d) in enumerate(datos, 1):
        
        z = (w1 * x1) + (w2 * x2) + b
        y = z  

        error = d - y
       
        if opcion == '1':
            perdida = error
        elif opcion == '2':
            perdida = 0.5*error**2
        elif opcion == '3':
            perdida = 0.25 * error**2

        # Actualización de pesos
        w1 = w1 + n * error * x1
        w2 = w2 + n * error * x2
        b = b + n * error

        
        print(f"\nPaso {paso}")
        print(f"Entrada: x1={x1}, x2={x2}, deseado={d}")
        print(f"Salida (y=z): {y:.4f}")
        print(f"(y - d): {error:.4f}")
        if opcion == '2':
            print(f"(y - d)^2: {perdida:.4f}")
        elif opcion == '3':
            print(f"1/4 * (y - d)^2: {perdida:.4f}")
        else:
            print(f"Pérdida: {perdida:.4f}")
        print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")
        print("-" * 30)

    # Evaluar al final de cada época
    def predict(X, w, b):
        z = np.dot(X, w) + b
        return np.where(z >= 0.5, 1, 0)

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    deseado = np.array([0, 1, 1, 0])
    w_final = np.array([w1, w2])
    b_final = b
    predicciones = predict(X, w_final, b_final)

    print("\n--- Evaluación tras la época ---")
    for i in range(4):
        print(f"Entrada: {X[i]}, Esperado: {deseado[i]}, Predicción: {predicciones[i]}")

    # Condición de parada anticipada
    if np.array_equal(predicciones, deseado):
        print("\n¡Entrenamiento detenido temprano! El modelo aprendió correctamente.")
        break


print("\n=== EVALUACIÓN FINAL ===")
for i in range(4):
    print(f"Entrada: {X[i]}, Salida esperada: {deseado[i]}, Predicción: {predicciones[i]}")


