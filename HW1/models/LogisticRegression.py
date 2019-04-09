import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W
        print("x.shape {}, y.shape{}".format(x.shape,y.shape))
        print(int(x.shape[0]/batch_size))
        
        for i in range(epochs):
            for j in range( int(x.shape[0]/batch_size) ):
                x_batch = x[batch_size * j: batch_size * (j+1)]
                y_batch = y[batch_size * j: batch_size * (j+1)]
                z = np.dot(x_batch, w)
                h = LogisticRegression._sigmoid(self, z)
                final_loss = (1/batch_size)*sum(-y_batch * np.log(h) - ( 1-y_batch)*np.log(1-h))
                wd = (1/batch_size)*(np.dot(np.transpose(x_batch), h-y_batch))
                w = optim.update(w, wd, lr)

        self.W = w
        print ("cost {}, batch_size {}, epoch {}".format(final_loss,batch_size,epochs))
        
        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        pred = np.zeros((x.shape[0], 1))
        h = LogisticRegression._sigmoid(self, np.dot(x, self.W))
        for k in range(h.shape[0]):
            if h[k] >= threshold:
                pred[(k,0)] = 1
            else:
                pred[(k,0)] = 0
        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 + np.exp(-x))
        # ============================================================
        return sigmoid