#%% test
from numbers import Real
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optuna as op

from optuna.pruners import BasePruner
from optuna.trial import TrialState

GridValueType = Union[str, float, int, bool, None]

_logger = get_logger(__name__)

class PredefinedSampler(BaseSampler):
    """Sampler modified from GridSampler
    """
    def __init__(
        self, search_space: Mapping[str, Sequence[GridValueType]], samp_list, seed: Optional[int] = None
    ) -> None:
        for param_name, param_values in search_space.items():
            for value in param_values:
                self._check_value(param_name, value)

        self._search_space = {}
        for param_name, param_values in (search_space.items()):
            self._search_space[param_name] = list(param_values)

        # self._all_grids = list(itertools.product(*self._search_space.values()))
        self._all_grids = samp_list
        self._visit_grids = []
        self._param_names = list(search_space.keys())
        self._n_min_trials = len(self._all_grids)
        self._rng = np.random.RandomState(seed)

    def reseed_rng(self) -> None:
        self._rng.seed()


    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        # Instead of returning param values, PredefinedSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        # When the trial is created by RetryFailedTrialCallback or enqueue_trial, we should not
        # assign a new grid_id.
        if "grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            print ("grid_id in trial.system_attrs")
            return {}

        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            
            # This case may occur with distributed optimization or trial queue. If there is no
            # target grid, `PredefinedSampler` evaluates a visited, duplicated point with the current
            # trial. After that, the optimization stops.

            _logger.warning(
                "`PredefinedSampler` is re-evaluating a configuration because the grid has been "
                "exhausted. This may happen due to a timing issue during distributed optimization "
                "or when re-running optimizations on already finished studies."
            )
            
            # One of all grids is randomly picked up in this case.
            target_grids = list(range(len(self._all_grids)))

        # In distributed optimization, multiple workers may simultaneously pick up the same grid.
        # To make the conflict less frequent, the grid is chosen randomly.
        grid_id = int(self._rng.choice(target_grids))
                    

        study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

        return {}


    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}


    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if "grid_id" not in trial.system_attrs:
            message = "All parameters must be specified when using PredefinedSampler with enqueue_trial."
            raise ValueError(message)

        if param_name not in self._search_space:
            message = "The parameter name, {}, is not found in the given grid.".format(param_name)
            raise ValueError(message)

        # TODO(c-bata): Reduce the number of duplicated evaluations on multiple workers.
        # Current selection logic may evaluate the same parameters multiple times.
        # See https://gist.github.com/c-bata/f759f64becb24eea2040f4b2e3afce8f for details.
        grid_id = trial.system_attrs["grid_id"]
        # print("run grid ", grid_id)
        param_value = self._all_grids[grid_id][self._param_names.index(param_name)]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value


    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        target_grids = self._get_unvisited_grid_ids(study)
        # if len(target_grids)%50==0:
        #     print("left sample: ",len(target_grids) )
        if len(target_grids) == 0:
            study.stop()
        elif len(target_grids) == 1:
            grid_id = study._storage.get_trial_system_attrs(trial._trial_id)["grid_id"]
            if grid_id == target_grids[0]:
                study.stop()


    @staticmethod
    def _check_value(param_name: str, param_value: Any) -> None:
        if param_value is None or isinstance(param_value, (str, int, float, bool)):
            return

        message = (
            "{} contains a value with the type of {}, which is not supported by "
            "`PredefinedSampler`. Please make sure a value is `str`, `int`, `float`, `bool`"
            " or `None` for persistent storage.".format(param_name, type(param_value))
        )
        warnings.warn(message)

    def _get_unvisited_grid_ids(self, study: Study) -> List[int]:
        # List up unvisited grids based on already finished ones.
        visited_grids = []
        running_grids = []

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for t in trials:
            if "grid_id" in t.system_attrs and self._same_search_space(
                t.system_attrs["search_space"]
            ):
                if t.state.is_finished():
                    visited_grids.append(t.system_attrs["grid_id"])
                elif t.state == TrialState.RUNNING or t.state == TrialState.WAITING:
                    running_grids.append(t.system_attrs["grid_id"])

        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids) - set(running_grids)
        # print(f'visited_grids {visited_grids} running_grids {running_grids}')

        # If evaluations for all grids have been started, return grids that have not yet finished
        # because all grids should be evaluated before stopping the optimization.
        if len(unvisited_grids) == 0:
            unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)

    @staticmethod
    def _grid_value_equal(value1: GridValueType, value2: GridValueType) -> bool:
        value1_is_nan = isinstance(value1, Real) and np.isnan(float(value1))
        value2_is_nan = isinstance(value2, Real) and np.isnan(float(value2))
        return (value1 == value2) or (value1_is_nan and value2_is_nan)

    def _same_search_space(self, search_space: Mapping[str, Sequence[GridValueType]]) -> bool:
        if set(search_space.keys()) != set(self._search_space.keys()):
            return False

        for param_name in search_space.keys():
            if len(search_space[param_name]) != len(self._search_space[param_name]):
                return False

            for i, param_value in enumerate(search_space[param_name]):
                if not self._grid_value_equal(param_value, self._search_space[param_name][i]):
                    return False

        return True
