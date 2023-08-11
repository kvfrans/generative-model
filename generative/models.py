import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable

activation = nn.tanh

class GeneratorLinear(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: N-length vector.
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = nn.Dense(features=128)(x)
        x = activation(x)
        x = nn.Dense(features=28*28*3)(x)
        x = nn.sigmoid(x)
        # Output: 28x28x3 image.
        x = x.reshape(x.shape[0], 28, 28, 3)
        return x
    
class EncoderLinear(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: 28*28*3 image.
        x = x.reshape(x.shape[0], 28*28*3)
        x = nn.Dense(features=128)(x)
        x = activation(x)
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = nn.Dense(features=32)(x)
        x = x.reshape(x.shape[0], 16, 2)
        # Output: 16*2 length vector of mean and std.
        mu = x[:, :, 0]
        logvar = x[:, :, 1]
        std = jnp.exp(0.5 * logvar)
        return mu, std
    
class VAE(nn.Module):
    def setup(self):
        self.encoder = EncoderLinear()
        self.decoder = GeneratorLinear()

    def __call__(self, x):
        mean, std = self.encoder(x)
        rng = self.make_rng('noise')
        x = jax.random.normal(rng, shape=mean.shape) * std + mean
        x = self.decoder(x)
        return x, mean, std
    
    def decode(self, z):
        return self.decoder(z)
    
class MiniBatchDiscrimination(nn.Module):
    out_features: int
    num_kernels: int = 4
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        T  = self.param('T', self.kernel_init, (jnp.shape(x)[-1], self.out_features * self.num_kernels))
        m = jnp.dot(x, T)
        m = m.reshape((jnp.shape(x)[0], self.out_features, self.num_kernels)) # [Batch, Out, NumKernels]
        m = jnp.expand_dims(m, 0) # [1, Batch, Out, NumKernels]
        mT = jnp.transpose(m, axes=(1, 0, 2, 3)) # [Batch, 1, Out, NumKernels]
        dists = jnp.abs(m - mT) # [Batch, Batch, Out, NumKernels]
        sum_dists = jnp.sum(dists, axis=-1) # [Batch, Batch, Out]
        exp_sum_dists = jnp.exp(-sum_dists) # [Batch, Batch, Out]
        out = jnp.sum(exp_sum_dists, axis=0) - 1 # [Batch, Out]

        x = jnp.concatenate([x, out], axis=-1)
        return x
    
class DiscriminatorLinear(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: 28*28*3 image.
        x = x.reshape(x.shape[0], 28*28*3)
        x = nn.Dense(features=128)(x)
        x = activation(x)
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = MiniBatchDiscrimination(out_features=8)(x)
        x = nn.Dense(features=32)(x)
        x = activation(x)
        x = nn.Dense(features=1)(x)
        # Output:
        return x
    
class GeneratorRNN(nn.Module):

    @nn.compact
    def __call__(self, x): # [Batch, Time, 5]
        x = nn.Dense(features=32)(x)
        x = activation(x)
        x = nn.Dense(features=64)(x)
        x = activation(x)
        # It automatically expects time to be after batch.
        lstm = nn.RNN(
            nn.LSTMCell(32), return_carry=True, name='encoder'
        )
        carry, x = lstm(x)
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = nn.Dense(features=512)(x) # [Batch, Time, 512]. Logits
        return x
    
class GeneratorRNNFlat(nn.Module):

    @nn.compact
    def __call__(self, x): # [Batch, Time * 8]
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = nn.Dense(features=64)(x)
        x = activation(x)
        x = nn.Dense(features=512)(x) # [Batch, Time, 512]. Logits
        return x
    
class DiffusionLinear(nn.Module):

    @nn.compact
    def __call__(self, x, time):
        # X: [B, 28, 28, 3]
        # Time: [B, 3]
        input = x
        x = x.reshape(x.shape[0], 28*28*3)
        x = nn.Dense(features=256)(x)
        x = jnp.concatenate([x, time], axis=-1)
        x = activation(x)
        x = nn.Dense(features=256)(x)
        x = activation(x)
        x = nn.Dense(features=256)(x)
        x = activation(x)
        x = nn.Dense(features=28*28*3)(x)
        x = x.reshape(x.shape[0], 28, 28, 3)
        return x
        # output = input+x
        # return output