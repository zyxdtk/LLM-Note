import torch

# 构造训练数据
torch.manual_seed(42)
x = torch.randn(100, 2)
true_w = torch.tensor([[3.0], [4.0]])
true_b = 5.0 

# 加入噪声
y = torch.matmul(x, true_w) + true_b + torch.randn(100, 1) * 0.1 

w = torch.randn(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr  = 0.1 
epochs = 100 

for epoch in range(epochs):
    y_pred = torch.matmul(x, w) + b 
    loss = torch.mean((y-y_pred)**2)
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad 
        b -= lr * b.grad 
        w.grad.zero_()
        b.grad.zero_()
    if (epoch % 10 == 0):
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print(f"Final w: {w.flatten()}")
print(f"Final b: {b.item()}")
