class SigmoidNeuron:

  def __init__(self):
    self.w = None
    self.b = None
  
  def perceptron(self,x):
    return np.dot(x,self.w.T) + self.b

  def sigmoid(self,x):
    return 1.0 / (1.0 + np.exp(-x))

  def grad_w(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred) * x

  def grad_b(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred)

  def fit(self, X, Y, epoch = 1, learning_rate = 1, initialize = True, Display_loss = False):

    if initialize:
      self.w = np.random.randn(1, X.shape[1])
      self.b = 0

    if Display_loss:
      loss = {}
    
    for i in tqdm_notebook(range(epoch), total = epoch, unit = 'epoch'):

      dw = 0
      db = 0

      for x, y in zip(X,Y):

        dw += self.grad_w(x,y)
        db += self.grad_b(x,y)

      self.w -= learning_rate * dw
      self.b -= learning_rate * db

      if Display_loss:
        y_pred = self.sigmoid(self.perceptron(X))
        loss[i] = mean_squared_error(y_pred,Y)

    if Display_loss:
      plt.plot(list(loss.values()))
      plt.xlabel("epoch")
      plt.ylabel("error")
      plt.show()
  def predict(self,X):
    y_pred = []
    
    for x in X:
      y_pred.append(self.sigmoid(self.perceptron(x)))
    
    return np.array(y_pred)
