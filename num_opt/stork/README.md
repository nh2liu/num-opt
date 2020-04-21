# Stork

Stork is a my neural network implementation from scratch that imitates pytorch's approach to computational graphs.

It supports an api using the `Sequential()` class, similar to pytorch.

Instead of `torch.Tensor`, the `Npw` class, numpy wrapper is responsible of propagating all numpy functions to the container object
to avoid defining each of the `torch.math_fn` functions.

Here's some sample code:

```python
model = Sequential(
    Linear(784, 150),
    Relu(),
    Linear(150, 10),
    Softmax(),
)

opt = Adam(model.parameters(), lr = 0.01)
loss_fn = CrossEntropyLoss()
losses = []

for i in range(1000):
    ...
    y_pred = model(x_batch)
    loss = loss_fn(y_pred, y_batch)
    loss.backward()
    losses.append(loss.item())
    opt.step()

```