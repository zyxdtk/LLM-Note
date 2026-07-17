import torch 
import torch.nn as nn
import torch.optim as optim

x = torch.randn(100, 2)
true_w = torch.tensor([[2.0], [3.0]])
y = torch.matmul(x, true_w) + 5.0 + torch.randn(100, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_dim=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(x)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

print(f"Final w: {model.linear.weight.data}")
print(f"Final b: {model.linear.bias.data}")
