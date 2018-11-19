import abc
from torch import nn
import copy


class TaskBase(nn.Module, abc.ABC):
    """
    abstract base class for tasks
    for each task, inherit and overwrite the methods
    note the type of the return value depends on the state dimension

    """

    def __init__(self, env, state_dim, batch_size=1):
        super(TaskBase, self).__init__()
        self.init_env = copy.copy(env)
        self.env = [copy.copy(env) for i in range(batch_size)]
        self.state_dim = state_dim
        self.batch_size = batch_size

    def reset(self):
        self.env = [copy.copy(self.init_env) for i in range(self.batch_size)]

    @abc.abstractmethod
    def f_enc(self, args):
        """encoder fn:: (args: Tensor[args_dim]) -> Tensor[state_dim]"""

    @abc.abstractmethod
    def f_env(self, prog_id, args):
        """env fn:: (args: Tensor[args_dim], prog_id: int)"""
        """change env inside"""
