
import chainer
import chainer.functions as F
import chainer.links as L

# network model
class Road_multi(chainer.Chain):

    def __init__(self):
        super(Road_multi, self).__init__()
	# init network
        with self.init_scope():      
            self.conv1=L.Convolution2D(3, 64, 16, stride=4, pad=0)
            self.conv2=L.Convolution2D(64, 112, 4, stride=1, pad=0)
            self.conv3=L.Convolution2D(112, 80, 3, stride=1, pad=0)
            self.fc4=L.Linear(3920, 4096)
            self.fc5=L.Linear(4096, 768)
    # forword process
    # __call_ is the same as forward
    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 1)
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.dropout(F.relu(self.fc4(h))) # dropout
        h = self.fc5(h)
        h = F.reshape(h, (-1, 3, 16, 16)) 
        if chainer.config.train:
            return h
        else:
            return F.softmax(h) # predict multi-class classification

model = Road_multi()
