from .transformer import create_transformer
from .lstm import create_lstm
from .cnn import create_cnn

models = dict(transformer=create_transformer,
              lstm=create_lstm,
              cnn=create_cnn
              )


def get_model(model_name, **kwargs):
    return models[model_name](**kwargs)
