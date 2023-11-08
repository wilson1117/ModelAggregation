

def align_neuron(model, indices):
    model.weight.data = model.weight[indices]
    model.bias.data = model.bias[indices]

def align_normalize(bn, indices):
    bn.weight.data = bn.weight[indices]
    bn.bias.data = bn.bias[indices]
    bn.running_var.data = bn.running_var[indices]
    bn.running_mean.data = bn.running_mean[indices]