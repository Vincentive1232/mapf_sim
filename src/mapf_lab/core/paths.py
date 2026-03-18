from __future__ import annotations

"""Path container types used by planners and coordination modules."""

from dataclasses import dataclass, field

import numpy as np

@dataclass
class DiscretePath:
    """Discrete path represented as a sequence of states.

    Attributes:
        states: Sequence of robot states along the path.
    """

    states: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of states in the path.

        Returns:
            Number of stored states.
        """
        return len(self.states)

    def __getitem__(self, t: int) -> np.ndarray:
        """Return the state at time step ``t``.

        Args:
            t: Non-negative time index.

        Returns:
            State at index ``t``. For indices beyond the last state, returns
            the final state (agent waits at goal).

        Raises:
            IndexError: If the path is empty or ``t`` is negative.
        """
        if not self.states:
            raise IndexError("Path is empty")
        if t < 0:
            raise IndexError("Negative time index is not supported")
        if t >= len(self.states):
            # For time steps beyond the end of the path, we assume the robot remains at the last state, 
            # basic assumption for CBS and similar algorithms where agents wait at their goal after reaching it.
            return self.states[-1]  # return the last state for out-of-bounds indices
        return self.states[t]

    def cost(self) -> float:
        """Return the discrete path cost measured in transition steps.

        Returns:
            ``inf`` when the path is empty, otherwise ``len(states) - 1``.
        """
        if not self.states:
            return float("inf")
        return float(max(0, len(self.states) - 1))

    def to_list(self) -> list[list[float]]:
        """Return the path states converted to nested Python lists.

        Returns:
            Path as ``list[list[float]]``.
        """
        return [s.tolist() for s in self.states]