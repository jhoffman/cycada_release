import torch 
from torch.autograd import Variable

def make_variable(tensor, volatile=False, requires_grad=True):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    if volatile:
        requires_grad = False
    return Variable(tensor, volatile=volatile, requires_grad=requires_grad)

