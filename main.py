from nn_layer import NNLayers

# let's try to build a simple neural network and train it with GD
nin = 3
nouts = [4, 4, 1]
x = [[2.0, 3.0, -1.0],
     [3.0, -1.0, 0.5],
     [0.5, 1.0, 1.0],
     [1.0, 1.0, -1.0]]
y = [1.0, -1.0, -1.0, 1.0]

nnlayers = NNLayers(nin=nin, nouts=nouts)

lr = 0.05
epochs = 200

for epoch in range(epochs):
    # forward:
    ypred = [nnlayers(xin)[0] for xin in  x]
    # compute loss
    avg_loss = sum([(y_ - ypred_)**2 for y_, ypred_ in zip(y, ypred)])/len(y)
    if epoch % 5 == 0:
        print (avg_loss)
    # zero the grad
    for p in nnlayers.parameters():
        p.grad = 0.0
    # compute gradient
    avg_loss.backward()
    # gradient descent
    for p in nnlayers.parameters():
        p.data -= lr*p.grad 

ypred = [nnlayers(xin)[0] for xin in  x]
print (f'ypred: {[ypred_.data for ypred_ in ypred]}')
print (f'y ground truth: {y}')