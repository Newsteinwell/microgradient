## here is a repo to reproduce the *micrograd* of Andrej Karpathy's github, which is basically the mini version of autograd in pytorch.

### some key notes:

- it shows how to defind a class (**Value**, see micrograd_engine.py) and with data, name (label), grad and some other attributes, which will be used to build a *micrograd*. Here we call it **engine**;
- in *micrograd*, you need defind the operators (sum, multiply, sub, div, exp) by yourself. and also you need define the corrsponding gradient of each operation. The most important formula is
  final_grad = local_grad $\times$ global_grad, which is actually the chain rule. With function `.backward()`, you can compute the gradient for each node from end to starting.
- it also shows how to build and optimize a simple neural network (MLP) with the object of **Value**. To run it, you can simply use
  ```bash
  python main.py
  ```
  Here we optimize the MLP by gradient descent manually, i.e. `param -= param.grad`.
- we are able to visualize the graph of our computation process or neural network with function `draw_dot` in file `utils.py`, it will show the name (label), data and gradient of each node. 
