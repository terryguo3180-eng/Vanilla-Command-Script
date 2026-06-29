from vcs import cmd
from vcs import cmdgen as cgn


# TODO: Not implemented yet
class DatapackPeepholeOptimizer:
    def __init__(self, cmdgen: cgn.CommandGenerator):
        self.cmdgen = cmdgen
    
    def optimize(self) -> cmd.Datapack | None:
        datapack = self.cmdgen.generate()
        if datapack is None:
            return None
        return self.process(datapack)
    
    def process(self, datapack: cmd.Datapack):
        return datapack
