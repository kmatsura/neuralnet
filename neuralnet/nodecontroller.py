from abc import ABCMeta, abstractmethod
from .node import AbstractNode, SimpleNode 


class NodeController():
    def add_nodes(self, nodeSet, adding_nodes):
        nodeSet.extend(adding_nodes)
        return nodeSet
