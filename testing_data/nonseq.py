from ..net import neural_net
from .quicktest_data import non_sequential
import numpy as np

x, y = non_sequential()

nn = neural_net(10)
nn.addlayer('dense', 10, 'sigmoid')
nn.addlayer('dense', 10, 'tanh')
nn.addlayer('dense', 10, 'sigmoid')

nn.fit(x, y, epochs=500, alpha=1, clip=0.25)

nn.reset()

res = nn.predict(x)
print(np.round(res))
