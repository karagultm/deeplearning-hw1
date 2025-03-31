import matplotlib.pyplot as plt
import numpy as np

# Muhammed Talha Karagül # 

# ----- part b ------
def gradient (x,y):
    df_dx = -2*x*np.exp(-(x**2+y**2)) - 3*2*2*(x-2)*np.exp(-2*((x-2)**2+(y-3)**2)) + 4*2*2*(x+4)*np.exp(-2*((x+4)**2+(y+3)**2))
    df_dy = -2*y*np.exp(-(x**2+y**2)) - 3*2*2*(y-3)*np.exp(-2*((x-2)**2+(y-3)**2)) + 4*2*2*(y+3)*np.exp(-2*((x+4)**2+(y+3)**2))

    return df_dx, df_dy

def gradient_ascent(points, learning_rate=0.01, num_iterations=100):
    paths = []
    for point in points:
        x, y = point
        path = [(x, y)]
        for _ in range(num_iterations):
            dx, dy = gradient(x, y)
            x += learning_rate * dx
            y += learning_rate * dy
            path.append((x, y))
        paths.append(path)
    return paths
# -----------------------

def gradient_descent(points, learning_rate=0.01, num_iterations=100):
    paths = []
    for point in points:
        x, y = point
        path = [(x, y)]
        for _ in range(num_iterations):
            dx, dy = gradient(x, y)
            x -= learning_rate * dx
            y -= learning_rate * dy
            path.append((x, y))
        paths.append(path)
    return paths

np.random.seed(42)  # For reproducibility

# ----- part a ------
# x ve y aralıklarını belirle
x = np.linspace(-5, 5, 100)  # X ekseni
y = np.linspace(-5, 5, 100)  # Y ekseni

# Meshgrid oluştur
X, Y = np.meshgrid(x, y)

# Fonksiyon değerlerini hesapla - Burada düzeltme yaptım
Z = np.exp(-(X**2+Y**2)) + 3 * np.exp(-2*((X-2)**2+(Y-3)**2)) - 4 * np.exp(-2*((X+4)**2+(Y+3)**2)) 

# Contour çizdir
plt.contour(X, Y, Z, levels=20, cmap="viridis")  

# Eksen etiketleri
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour Plot of Function")
plt.savefig('question1_function.png')

# Renk çubuğu ekle
plt.colorbar(label="Z values")


# -----------------------

random_points = [(np.random.uniform(-5, 5), np.random.uniform(-5, 5)) for _ in range(5)]

colors = ['r', 'g', 'b', 'm', 'c']  # Farklı renkler
markers = ['o', 's', '^', 'D', 'p']  # Farklı işaretler

# ------ part c --------- 
ascent_paths = gradient_ascent(random_points, learning_rate=0.1, num_iterations=100)
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour, label='Function value')
plt.title('Gradient Ascent Paths')
plt.xlabel('x')
plt.ylabel('y')

for i, path in enumerate(ascent_paths):
    path_array = np.array(path)

    # Yol boyunca oklar çiz
    for j in range(len(path_array) - 1):
        plt.quiver(path_array[j, 0], path_array[j, 1],
                   path_array[j+1, 0] - path_array[j, 0],
                   path_array[j+1, 1] - path_array[j, 1],
                   angles='xy', scale_units='xy', scale=1, color=colors[i % len(colors)])

    # Başlangıç noktası
    plt.plot(path_array[0, 0], path_array[0, 1], color=colors[i % len(colors)], marker='o', markersize=8, label=f'Start {i+1}')
    
    # Bitiş noktası
    plt.plot(path_array[-1, 0], path_array[-1, 1], color=colors[i % len(colors)], marker='x', markersize=8, label=f'End {i+1}')

plt.legend()
plt.grid(True)
plt.savefig('question1_gradient_ascent.png')


#------------------------

descent_paths = gradient_descent(random_points, learning_rate=0.1, num_iterations=100)
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour, label='Function value')
plt.title('Gradient Descent Paths')
plt.xlabel('x')
plt.ylabel('y')

for i, path in enumerate(descent_paths):
    path_array = np.array(path)

    # Yol boyunca oklar çiz
    for j in range(len(path_array) - 1):
        plt.quiver(path_array[j, 0], path_array[j, 1],
                   path_array[j+1, 0] - path_array[j, 0],
                   path_array[j+1, 1] - path_array[j, 1],
                   angles='xy', scale_units='xy', scale=1, color=colors[i % len(colors)])

    # Başlangıç noktası
    plt.plot(path_array[0, 0], path_array[0, 1], color=colors[i % len(colors)], marker='o', markersize=8, label=f'Start {i+1}')
    
    # Bitiş noktası
    plt.plot(path_array[-1, 0], path_array[-1, 1], color=colors[i % len(colors)], marker='x', markersize=8, label=f'End {i+1}')

plt.legend()
plt.grid(True)
plt.savefig('question1_gradient_descent.png')




plt.show()

plt.close('all')