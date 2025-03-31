import numpy as np
import matplotlib.pyplot as plt

# Veri seti
X = np.array([10, 15, 20, 40, 50, 60, 60, 70, 80, 90, 95, 100, 100])
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])

# Sigmoid fonksiyonu
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent ile parametreleri bulma
def logistic_regression(X, y, learning_rate=0.01, epochs=10000):
    # Başlangıç parametreleri
    beta_0 = 0.0
    beta_1 = 0.0
    n = len(X)
    
    # Gradient Descent
    for _ in range(epochs):
        # Tahmin
        z = beta_0 + beta_1 * X
        y_pred = sigmoid(z)
        
        # Gradient'ları hesapla
        error = y_pred - y
        grad_beta_0 = np.sum(error) / n
        grad_beta_1 = np.sum(error * X) / n
        
        # Parametreleri güncelle
        beta_0 -= learning_rate * grad_beta_0
        beta_1 -= learning_rate * grad_beta_1
    
    return beta_0, beta_1

# Modeli eğit
beta_0, beta_1 = logistic_regression(X, y)
print(f"beta_0: {beta_0}")
print(f"beta_1: {beta_1}")

# Modeli görselleştirme
X_plot = np.linspace(min(X)-10, max(X)+10, 100)
P = sigmoid(beta_0 + beta_1 * X_plot)

plt.scatter(X, y, color='blue', label='Veri')
plt.plot(X_plot, P, color='red', label=f'Logistic Fit\nβ0={beta_0:.2f}, β1={beta_1:.4f}')
plt.xlabel('X')
plt.ylabel('P(y=1|X)')
plt.legend()
plt.title('Logistic Regression Model')
plt.savefig('question3.png')
plt.show()