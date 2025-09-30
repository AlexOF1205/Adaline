n = 0.1
y = 0
w1 = 0.1
w2 = 0.1
b = 0.2
x1 = 0
x2 = 0
d = 0
fila  = 1
x = 1
epoca = 1   
while x <= 100:
    if fila == 5:
        fila = 1

#Modificable (esto representa la tabla)
    if fila == 1:
        print("Época", epoca)
        epoca += 1
        x1 = 0
        x2 = 0
        d = 0
    if fila == 2:
        x1 = 0
        x2 = 1
        d = 1
    if fila == 3:
        x1 = 1
        x2 = 0
        d = 1
    if fila == 4:
        x1 = 1
        x2 = 1
        d = 0

        
    z = (w1 * x1) + (w2 * x2) + b
    
    error = 1/4*(d - z)*(d - z)
    if error != 0:
        w1 = w1 + n * error * x1
        w2 = w2 + n * error * x2
        b = b + n * error

    
    print("Paso ", x)
    print("w1 = ", w1)
    print("w2 = ", w2)
    print("b = ", b)
    print("----------------")
    x += 1
    fila += 1
