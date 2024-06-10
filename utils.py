from graphviz import Digraph

def trace(root):
    # build a set of nodes and edges of a computation grpah
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            # if isinstance(v, Value):
            if True:
                nodes.add(v)
                # print (f'value: {v}, label: {v.label}')
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR= left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any node in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this node is created by some operations, create an op for it
            dot.node(name=uid+n._op, label=n._op)
            # connect this node to it
            dot.edge(uid+n._op, uid)
        
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2))+n2._op)
    
    return dot