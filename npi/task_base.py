import abc


class TaskBase(abc.ABC):
    """
    abstract base class for tasks
    for each task, inherit and overwrite the methods
    note the type of the return value depends on the state dimension

    """

    @abc.abstractmethod
    def __init__(self, env, state_dim):
        self.env = env
        self.state_dim = state_dim

    @abc.abstractmethod
    def f_enc(self, args):
        """encoder fn:: (Env, args: Tensor[args_dim]) -> Tensor[state_dim]"""

    @abc.abstractmethod
    def f_env(self, prog_id, args):
        """env fn:: (Env, args: Tensor[args_dim], prog_id: int) -> Env"""
