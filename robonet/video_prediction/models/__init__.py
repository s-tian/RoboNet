from .graphs import get_graph_class


def get_model(class_name):
    if class_name == 'deterministic':
        from .deterministic_generator import DeterministicModel
        return DeterministicModel
    elif class_name == 'stochastic':
        from .stochastic_generator import StochasticModel 
        return StochasticModel 
    else:
        raise NotImplementedError
