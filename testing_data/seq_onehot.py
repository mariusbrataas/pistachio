from ..net import neural_net
from .quicktest_data import seq
import numpy as np

raw = input('String to learn: ')
if raw == '': raw = 'fiskeost i brun saus'

let2num, num2let = {}, {}

for letter in raw:
    if not letter in let2num: let2num[letter] = len(let2num)
for letter in let2num:
    num2let[let2num[letter]] = letter

print(let2num)
print(num2let)

data = []
for letter in raw:
    tmp = np.zeros(len(let2num))
    tmp[let2num[letter]] = 1
    data.append(tmp)

x = data[0:-1]
y = data[1:]

nn = neural_net(len(let2num))
nn.add('recurrent', 10, 'sigmoid')
nn.add('recurrent', 10, 'tanh')
nn.add('recurrent', len(let2num), 'sigmoid')

nn.fit_sequence(x, y, epochs=500, alpha=1, clip=0.25)

nn.reset()

res = [num2let[let2num[raw[0]]]]
for tmp in x:
    res.append(num2let[np.argmax(nn.predict(tmp))])

print(res)
