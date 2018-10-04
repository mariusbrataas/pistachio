from ..net import neural_net
from .quicktest_data import seq
import numpy as np

x, y = seq()

nn = neural_net(10)
nn.addlayer('recurrent', 10, 'sigmoid')
nn.addlayer('recurrent', 10, 'tanh')
nn.addlayer('recurrent', 10, 'sigmoid')

nn.fit_sequence(x, y, epochs=500, alpha=1, clip=0.25)

nn.reset()

#res = nn.predict(x)
print(np.round(nn.predict(x[0])))
print(np.round(nn.predict(x[0])))
