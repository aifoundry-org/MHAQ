from torch import nn

# Apparently in some architectures one intend to access 
# the weight of the module directly invocating "module.weight" 
# during the forward pass.
# And since we are wrapping module into a Sequential container 
# we need to pass weights of the module directly outside of the container.
class NSequential(nn.Sequential):
    @property
    def weight(self):
        return self._modules["0"].weight