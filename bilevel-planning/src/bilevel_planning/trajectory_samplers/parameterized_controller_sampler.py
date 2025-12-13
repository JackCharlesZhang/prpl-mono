"""A trajectory sampler that uses parameterized controllers with parameter samplers."""

from typing import Callable, Hashable, TypeVar

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import ParameterizedController, TransitionFailure
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
    TrajectorySamplingFailure,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class ParameterizedControllerTrajectorySampler(TrajectorySampler[_X, _U, _S, _A]):
    """A trajectory sampler that uses parameterized controllers + parameter samplers."""

    def __init__(
        self,
        controller_generator: Callable[[_A], ParameterizedController[_X, _U]],
        transition_function: Callable[[_X, _U], _X],
        state_abstractor: Callable[[_X], _S],
        max_trajectory_steps: int,
    ) -> None:
        self._controller_generator = controller_generator
        self._transition_function = transition_function
        self._state_abstractor = state_abstractor
        self._max_trajectory_steps = max_trajectory_steps

    def __call__(
        self,
        x: _X,
        s: _S,
        a: _A,
        ns: _S,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
        rng: np.random.Generator,
    ) -> tuple[list[_X], list[_U]]:
        """Samples a trajectory or raises TrajectorySamplingFailure().

        Updates bpg in-place with every transition sampled.
        """
        # Get the parameterized controller.
        controller = self._controller_generator(a)

        # Initialize the trajectory.
        x_traj: list[_X] = [x]
        u_traj: list[_U] = []

        # Reset the controller.
        params = controller.sample_parameters(x, rng)

        # Sample parameters for the controller.
        controller.reset(x, params)

        # Simulate until termination.
        transition_failed = False
        controller_terminated_early = False
        for step_num in range(self._max_trajectory_steps):
            # Break when the controller determines that it is done.
            if controller.terminated():
                controller_terminated_early = (step_num < self._max_trajectory_steps - 1)
                break
            # Get the next action.
            u = controller.step()
            # Move forward and terminate early upon transition failure.
            try:
                nx = self._transition_function(x, u)
            except TransitionFailure:
                transition_failed = True
                break
            # Update the controller.
            controller.observe(nx)
            # Update the trajectory.
            x_traj.append(nx)
            u_traj.append(u)
            # Update the graph.
            bpg.add_state_node(nx)
            bpg.add_action_edge(x, u, nx)
            # Advance the state.
            x = nx

        # Check if we succeeded in reaching the target abstract state.
        final_state = x_traj[-1]
        final_abstract_state = self._state_abstractor(final_state)
        bpg.add_abstract_state_node(final_abstract_state)
        bpg.add_state_abstractor_edge(final_state, final_abstract_state)

        # DEBUG: Log why this trajectory failed
        if final_abstract_state != ns:
            failure_reason = []
            if transition_failed:
                failure_reason.append("transition_failed")
            if controller_terminated_early:
                failure_reason.append(f"controller_terminated_early_at_step_{len(x_traj)-1}")
            if len(x_traj) >= self._max_trajectory_steps:
                failure_reason.append(f"max_steps_reached")
            if not failure_reason:
                failure_reason.append("wrong_final_abstract_state")

            print(f"  [TRAJ_FAIL] Action={a}, Steps={len(x_traj)-1}/{self._max_trajectory_steps}, "
                  f"Reason={','.join(failure_reason)}")
            print(f"    Expected: {ns}")
            print(f"    Got:      {final_abstract_state}")
            raise TrajectorySamplingFailure()

        # Success!
        print(f"  [TRAJ_SUCCESS] Action={a}, Steps={len(x_traj)-1}")
        return x_traj, u_traj
