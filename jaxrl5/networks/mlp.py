from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import flax

default_init = nn.initializers.xavier_uniform

def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()}))

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x