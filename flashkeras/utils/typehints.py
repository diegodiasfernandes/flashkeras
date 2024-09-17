from typing import overload, Union, Tuple, Literal, Any, Optional, cast
from keras.preprocessing.image import DirectoryIterator, NumpyArrayIterator # type: ignore
BatchIterator = Union[DirectoryIterator, NumpyArrayIterator]