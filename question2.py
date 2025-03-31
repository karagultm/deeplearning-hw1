import numpy as np
import matplotlib.pyplot as plt

# Veri seti
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([3, 7, 7, 11, 14, 21, 18, 18, 19, 23])

##### Part A - Linear Regression #####

a, b = 0, 0
learning_rate = 0.01
epochs = 1000
threshold = 0.0001

for i in range(epochs):
    y_pred = a * x + b
    error = y_pred - y
    cost = np.mean(error ** 2)

    da = (2 / len(x)) * np.sum(error * x)
    db = (2 / len(x)) * np.sum(error)
    
    a -= learning_rate * da
    b -= learning_rate * db
    
    if abs(da) < threshold and abs(db) < threshold:
        print(f"Early stopping at epoch {i}")
        break

print(f"Final coefficients: a = {a:.4f}, b = {b:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='True Data')
plt.plot(x, a * x + b, color='red', label=f'1st Degree Model: y = {a:.2f}x + {b:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('1st Degree')
plt.legend()
plt.grid(True)
plt.savefig('question2_first_order.png')

##### Part B - 10th Degree Polynomial Regression #####

degree = 10
learning_rate_2 = 0.0005
epochs = 10000
weights = np.zeros(degree+1)

x_mean = x.mean()
x_std = x.std()
y_mean = y.mean()
y_std = y.std() if y.std() != 0 else 1

x_norm = (x - x_mean) / x_std
y_norm = (y - y_mean) / y_std

def poly_predict(x, weights):
    return sum(w * (x ** i) for i, w in enumerate(weights))

def poly_grad(x, error, weights):
    grad = np.zeros_like(weights)
    for i in range(len(weights)):
        grad[i] = (2 / len(x)) * np.sum(error * (x ** i))
    return grad

for i in range(epochs):
    y_pred_norm = poly_predict(x_norm, weights)
    error = y_pred_norm - y_norm
    cost = np.mean(error ** 2)
  
    grad = poly_grad(x_norm, error, weights)
    weights -= learning_rate_2 * grad
    
    if i % 1000 == 0:
        print(f"Epoch {i}, Cost: {cost:.6f}, Gradient Norm: {np.linalg.norm(grad):.6f}")
    
    if np.any(np.isnan(weights)) or np.any(np.isnan(grad)):
        print("NaN detected! Stopping training.")
        break
    
    if np.all(np.abs(grad) < threshold):
        print(f"Early stopping at epoch {i}")
        break

print("10th degree final coefficients:")
for i in range(len(weights)):
    print(f"Weight {i}: {weights[i]:.4f}")

x_smooth = np.linspace(min(x), max(x), 100)
x_smooth_norm = (x_smooth - x_mean) / x_std
y_smooth_pred = poly_predict(x_smooth_norm, weights) * y_std + y_mean

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='True Data')
plt.plot(x_smooth, y_smooth_pred, color='red', label='10th Degree Model')

plt.xlabel('x')
plt.ylabel('y')
plt.title('10th Degree')
plt.legend()
plt.grid(True)
plt.savefig('question2_tenth_order.png')


##### Part C - 10th Degree Polynomial with Ridge Regularization #####

lambda_reg = 0.1  # Ridge regularization constant
epochs = 10000

weights = np.zeros(degree + 1)

def poly_grad_ridge(x, error, weights, lambda_reg):
    grad = np.zeros_like(weights)
    for i in range(len(weights)):
        grad[i] = (2 / len(x)) * np.sum(error * (x ** i)) + 2 * lambda_reg * weights[i]
    return grad

for i in range(epochs):
    y_pred_norm = poly_predict(x_norm, weights)
    error = y_pred_norm - y_norm
    cost = np.mean(error ** 2) + lambda_reg * np.sum(weights ** 2)
    
    grad = poly_grad_ridge(x_norm, error, weights, lambda_reg)
    weights -= learning_rate_2 * grad
    
    if i % 1000 == 0:
        print(f"Epoch {i}, Cost: {cost:.6f}, Gradient Norm: {np.linalg.norm(grad):.6f}")
    
    if np.all(np.abs(grad) < threshold):
        print(f"Early stopping at epoch {i}")
        break

print("Regularized 10th degree final coefficients:")
for i, w in enumerate(weights):
    print(f"Weight {i}: {w:.4f}")

x_smooth = np.linspace(min(x), max(x), 100)
x_smooth_norm = (x_smooth - x_mean) / x_std
y_smooth_pred = poly_predict(x_smooth_norm, weights) * y_std + y_mean

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='True Data')
plt.plot(x_smooth, y_smooth_pred, color='red', label=f'10th Degree (Ridge, λ={lambda_reg})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('10th Degree Polynomial with Ridge Regularization')
plt.legend()
plt.grid(True)
plt.savefig('question2_tenth_order_ridge.png')
plt.show()
