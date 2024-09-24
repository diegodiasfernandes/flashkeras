from .dataframes import load_wine, load_breast_cancer, load_diabetes, load_iris, boston_housing_load_data
from .image import load_cifar100, load_mnist, load_cifar10, load_fashion_mnist, load_digits
from .matrix_like import load_imdb, imdb_get_world_index, load_reuters, reuters_get_word_index, reuters_get_label_names

__all__ = [
    'load_wine', 
    'load_breast_cancer', 
    'load_diabetes', 
    'load_iris', 
    'load_cifar100',
    'load_mnist',
    'load_cifar10',
    'load_fashion_mnist',
    'load_digits',
    'load_imdb',
    'imdb_get_world_index',
    'load_reuters',
    'reuters_get_word_index',
    'reuters_get_word_index',
    'reuters_get_label_names',
    'boston_housing_load_data'
]