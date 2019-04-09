import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Training should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        
        # ========================= EDIT HERE ========================
        y = y.reshape(x.shape[0], 1)
        w = self.W
        print("x.shape {}, y.shape{}".format(x.shape,y.shape))
        print("num_epochs {}, lr {}".format(epochs, lr))
        n = self.num_features
        
        for i in range(epochs):
            for j in range( int(x.shape[0]/batch_size) ):
                x_batch = x[batch_size * j: batch_size * (j+1)]
                y_batch = y[batch_size * j: batch_size * (j+1)]
                y_predicted = np.dot(x_batch, w)
                loss = y_predicted-y_batch
                #print("x_batch\n{}, loss \n{}, x_batch*loss\n{}".format(x_batch, loss, x_batch*loss))
                final_loss = (1/batch_size) * sum([val**2 for val in (y_batch - y_predicted)])
                xT = np.transpose(x_batch)
                wd = (2/batch_size)*(np.dot(xT,loss))
                w = optim.update(w, wd, lr)
                #print ("wd {}, cost {}, j {}, iteration {}".format(wd,final_loss, j, i))


        self.W = w
        # ============================================================
        return final_loss

    def eval(self, x):
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        pred = np.dot(x , self.W)

        # ============================================================
        return pred
