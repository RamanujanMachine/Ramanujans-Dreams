"""
Gradient-ascent optimizer strategies.

Each optimizer consumes an estimated gradient ``g`` (a real-valued vector in the
flatland direction space) and returns the *update vector* to add to the current
real direction (before the learning-rate scaling and lattice snapping performed
by the caller).  Because the project maximizes delta, the optimizers ascend:
the update points along ``+g`` (Adam/RMSprop scale per-coordinate, momentum
smooths over steps).

The strategy pattern keeps each variant isolated and makes adding new optimizers
trivial — see :func:`optimizer_for`.
"""

from abc import ABC, abstractmethod

import numpy as np


class GradOptimizer(ABC):
    """Base class for gradient-ascent update rules over a fixed-dimension space."""

    def __init__(self, dim: int):
        """
        :param dim: Dimension of the gradient / update vectors.
        """
        self.dim = dim

    @abstractmethod
    def step(self, grad: np.ndarray) -> np.ndarray:
        """
        Consume a gradient estimate and return the (unscaled) ascent update vector.

        :param grad: Estimated gradient of delta w.r.t. the direction (length ``dim``).
        :return: The update vector to add to the current real direction.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear any accumulated internal state (moments, velocity, step count)."""
        raise NotImplementedError


class VanillaGrad(GradOptimizer):
    """Plain gradient ascent — the update is the raw gradient."""

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Return the gradient unchanged as the ascent update.

        :param grad: Estimated gradient.
        :return: ``grad`` itself.
        """
        return np.asarray(grad, dtype=np.float64)

    def reset(self) -> None:
        """No internal state to clear (stateless optimizer)."""
        return None


class Momentum(GradOptimizer):
    """
    Heavy-ball momentum: ``v <- beta*v + g``; update is ``v``.

    Smooths the (noisy, finite-difference) gradient across steps so the ascent
    keeps moving through small flat / non-differentiable regions.
    """

    def __init__(self, dim: int, beta: float = 0.9):
        """
        :param dim: Dimension of the update vectors.
        :param beta: Momentum coefficient in ``[0, 1)``.
        """
        super().__init__(dim)
        self.beta = beta
        self._v = np.zeros(dim, dtype=np.float64)

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Accumulate the gradient into the velocity and return it.

        :param grad: Estimated gradient.
        :return: The updated velocity ``v = beta*v + grad``.
        """
        self._v = self.beta * self._v + np.asarray(grad, dtype=np.float64)
        return self._v.copy()

    def reset(self) -> None:
        """Clear the accumulated velocity."""
        self._v = np.zeros(self.dim, dtype=np.float64)


class RMSprop(GradOptimizer):
    """
    RMSprop: per-coordinate scaling by the root mean square of recent gradients.

    ``s <- beta2*s + (1-beta2)*g^2``; update is ``g / (sqrt(s) + eps)``.
    """

    def __init__(self, dim: int, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        :param dim: Dimension of the update vectors.
        :param beta2: Decay rate of the squared-gradient running average.
        :param epsilon: Numerical-stability term in the denominator.
        """
        super().__init__(dim)
        self.beta2 = beta2
        self.epsilon = epsilon
        self._s = np.zeros(dim, dtype=np.float64)

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Scale the gradient per-coordinate by its running RMS.

        :param grad: Estimated gradient.
        :return: ``grad / (sqrt(s) + eps)`` with the updated second moment ``s``.
        """
        g = np.asarray(grad, dtype=np.float64)
        self._s = self.beta2 * self._s + (1.0 - self.beta2) * g * g
        return g / (np.sqrt(self._s) + self.epsilon)

    def reset(self) -> None:
        """Clear the squared-gradient running average."""
        self._s = np.zeros(self.dim, dtype=np.float64)


class Adam(GradOptimizer):
    """
    Adam: bias-corrected first and second moment estimates.

    ``m <- beta1*m + (1-beta1)*g``; ``s <- beta2*s + (1-beta2)*g^2``;
    update is ``m_hat / (sqrt(s_hat) + eps)`` with bias correction by step count.
    """

    def __init__(self, dim: int, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        :param dim: Dimension of the update vectors.
        :param beta1: First-moment decay rate.
        :param beta2: Second-moment decay rate.
        :param epsilon: Numerical-stability term in the denominator.
        """
        super().__init__(dim)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = np.zeros(dim, dtype=np.float64)
        self._s = np.zeros(dim, dtype=np.float64)
        self._t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Return the bias-corrected Adam update for the gradient.

        :param grad: Estimated gradient.
        :return: ``m_hat / (sqrt(s_hat) + eps)`` after updating both moments.
        """
        g = np.asarray(grad, dtype=np.float64)
        self._t += 1
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * g
        self._s = self.beta2 * self._s + (1.0 - self.beta2) * g * g
        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        s_hat = self._s / (1.0 - self.beta2 ** self._t)
        return m_hat / (np.sqrt(s_hat) + self.epsilon)

    def reset(self) -> None:
        """Clear both moment estimates and the step counter."""
        self._m = np.zeros(self.dim, dtype=np.float64)
        self._s = np.zeros(self.dim, dtype=np.float64)
        self._t = 0


class UnknownGradVariant(ValueError):
    """Raised when an unrecognised gradient-ascent variant name is requested."""

    def __init__(self, name: str):
        """
        :param name: The unrecognised variant name that was requested.
        """
        self.name = name
        super().__init__(
            f"Unknown gradient-ascent variant '{name}'. "
            f"Expected one of: 'vanilla', 'momentum', 'rmsprop', 'adam'."
        )


def optimizer_for(name: str, dim: int, cfg) -> GradOptimizer:
    """
    Build the optimizer strategy selected by *name* using config hyperparameters.

    :param name: Variant name ('vanilla' | 'momentum' | 'rmsprop' | 'adam'),
        case-insensitive.
    :param dim: Dimension of the gradient / update vectors.
    :param cfg: The search config object exposing ``GRAD_MOMENTUM``,
        ``GRAD_BETA1``, ``GRAD_BETA2``, ``GRAD_EPSILON``.
    :raise UnknownGradVariant: If *name* is not a recognised variant.
    :return: A fresh :class:`GradOptimizer` instance.
    """
    key = str(name).strip().lower()
    if key == "vanilla":
        return VanillaGrad(dim)
    if key == "momentum":
        return Momentum(dim, beta=cfg.GRAD_MOMENTUM)
    if key == "rmsprop":
        return RMSprop(dim, beta2=cfg.GRAD_BETA2, epsilon=cfg.GRAD_EPSILON)
    if key == "adam":
        return Adam(dim, beta1=cfg.GRAD_BETA1, beta2=cfg.GRAD_BETA2, epsilon=cfg.GRAD_EPSILON)
    raise UnknownGradVariant(name)
