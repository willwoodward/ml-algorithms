x = [0, 1, 2, 2, 3]
y = [5, 7, 8, 9, 11]
m = 5

def h(x_i, theta_0, theta_1):
    return theta_0 + (theta_1 * x_i)

def loss_gradient(theta_0, theta_1):
    sum = [0, 0]
    for i in range(m):
        sum[0] += h(x[i], theta_0, theta_1) - y[i]
        sum[1] += (h(x[i], theta_0, theta_1) - y[i]) * x[i]
    
    sum[0] = sum[0] / m
    sum[1] = sum[1] / m
    
    return sum

def training_loop(theta_0, theta_1, alpha, max_it):
    for _ in range(max_it):
        grad = loss_gradient(theta_0, theta_1)
        
        theta_0 = theta_0 - alpha * grad[0]
        theta_1 = theta_1 - alpha * grad[1]

    return theta_0, theta_1

theta_0, theta_1 = training_loop(1, 1, 0.01, 1000)

print(theta_0 + theta_1 * x[0])