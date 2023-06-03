"""
This module provides a simple lazy-loading mechanism for various libraries.
"""
from typing import Callable, cast, Generic, TypeVar
from types import ModuleType

Module = TypeVar("Module", bound=ModuleType)
class LazyModule(Generic[Module]):
    def __init__(self, importer: Callable[[], Module]):
        self.importer = importer
        self.module = None

    def __getattr__(self, attr):
        if self.module is None:
            self.module = self.importer()
        for k, v in globals().items():
            if v is self:
                globals()[k] = self.module
        return getattr(self.module, attr)

    # def __repr__(self):
    #     return f"LazyModule({self.importer()})"


def lazy_import(importer: Callable[[], Module]) -> Module:
    """
    Lazily-import a module while providing all available type information.

    The common structure looks like:

    ```py
    def __import_module():
        import module
        return module
    module = lazy_import(__import_module)
    ```

    When importing local modules, you must remove the module from the globals list:

    ```py
    def __import_local_module():
        del globals()["local_module"]
        from . import local_module
        return local_module
    local_module = lazy_import(__import_local_module)
    ```
    """
    return cast(Module, LazyModule(importer))


def __import_tensorflow():
    import tensorflow
    return tensorflow
tensorflow = lazy_import(__import_tensorflow)


def __import_wandb():
    import wandb
    return wandb
wandb = lazy_import(__import_wandb)
