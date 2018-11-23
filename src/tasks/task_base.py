import abc
from torch import nn
import copy


class TaskBase(nn.Module, abc.ABC):
    """
    abstract base class for tasks
    for each task, inherit and overwrite the methods
    note the type of the return value depends on the state dimension
    """

    def __init__(self, env, state_dim):
        """
        :param env: a list of initial env
        :param state_dim: dimension of state tensor
        """
        super(TaskBase, self).__init__()
        self.init_env = copy.copy(env)
        # it is a list in default
        self.env = env
        self.state_dim = state_dim
        self.batch_size = len(env)

    def reset(self):
        self.env = copy.copy(self.init_env)

    @abc.abstractmethod
    def f_enc(self, args):
        """
        encoder fn:: (args: Tensor[args_dim]) -> Tensor[state_dim]
        all args are in batches
        """

    @abc.abstractmethod
    def f_env(self, prog_id, args):
        """
        env fn:: (args: Tensor[args_dim], prog_id: int)
        all args are in batches
        change env inside
        """
