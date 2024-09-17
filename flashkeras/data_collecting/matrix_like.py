from keras.datasets import reuters, imdb  # type: ignore

def load_imdb():
    """Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

    This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
    (positive/negative). Reviews have been preprocessed, and each review is
    encoded as a list of word indexes (integers).
    For convenience, words are indexed by overall frequency in the dataset,
    so that for instance the integer "3" encodes the 3rd most frequent word in
    the data. This allows for quick filtering operations such as:
    "only consider the top 10,000 most
    common words, but eliminate the top 20 most common words".

    As a convention, "0" does not stand for a specific word, but instead is used
    to encode the pad token.

    Args:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: integer or None. Words are
          ranked by how often they occur (in the training set) and only
          the `num_words` most frequent words are kept. Any less frequent word
          will appear as `oov_char` value in the sequence data. If None,
          all words are kept. Defaults to `None`.
      skip_top: skip the top N most frequently occurring words
          (which may not be informative). These words will appear as
          `oov_char` value in the dataset. When 0, no words are
          skipped. Defaults to `0`.
      maxlen: int or None. Maximum sequence length.
          Any longer sequence will be truncated. None, means no truncation.
          Defaults to `None`.
      seed: int. Seed for reproducible data shuffling.
      start_char: int. The start of a sequence will be marked with this
          character. 0 is usually the padding character. Defaults to `1`.
      oov_char: int. The out-of-vocabulary character.
          Words that were cut out because of the `num_words` or
          `skip_top` limits will be replaced with this character.
      index_from: int. Index actual words with this index and higher.
      **kwargs: Used for backwards compatibility.

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train, x_test**: lists of sequences, which are lists of indexes
      (integers). If the num_words argument was specific, the maximum
      possible index value is `num_words - 1`. If the `maxlen` argument was
      specified, the largest possible sequence length is `maxlen`.

    **y_train, y_test**: lists of integer labels (1 or 0).

    Raises:
      ValueError: in case `maxlen` is so low
          that no input sequence could be kept.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    return imdb.load_data()

def imdb_get_world_index():
    """Retrieves a dict mapping words to their index in the IMDB dataset.

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).

    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.

    Example:

    ```python
    # Use the default parameters to keras.datasets.imdb.load_data
    start_char = 1
    oov_char = 2
    index_from = 3
    # Retrieve the training sequences.
    (x_train, _), _ = keras.datasets.imdb.load_data(
        start_char=start_char, oov_char=oov_char, index_from=index_from
    )
    # Retrieve the word index file mapping words to indices
    word_index = keras.datasets.imdb.get_word_index()
    # Reverse the word index to obtain a dict mapping indices to words
    # And add `index_from` to indices to sync with `x_train`
    inverted_word_index = dict(
        (i + index_from, word) for (word, i) in word_index.items()
    )
    # Update `inverted_word_index` to include `start_char` and `oov_char`
    inverted_word_index[start_char] = "[START]"
    inverted_word_index[oov_char] = "[OOV]"
    # Decode the first sequence in the dataset
    decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
    ```
    """
    return imdb.get_word_index()

def reuters_get_label_names():
    """Returns labels as a list of strings with indices matching training data.

    Reference:

    - [Reuters Dataset](https://martin-thoma.com/nlp-reuters/)
    """
    return reuters.get_label_names()

def reuters_get_word_index():
    """Retrieves a dict mapping words to their index in the Reuters dataset.

    Actual word indices starts from 3, with 3 indices reserved for:
    0 (padding), 1 (start), 2 (oov).

    E.g. word index of 'the' is 1, but the in the actual training data, the
    index of 'the' will be 1 + 3 = 4. Vice versa, to translate word indices in
    training data back to words using this mapping, indices need to substract 3.

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).

    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.
    """

    return reuters.get_word_index()

def load_reuters(test_split: float = 0.2):
    """Loads the Reuters newswire classification dataset.

    This is a dataset of 11,228 newswires from Reuters, labeled over 46 topics.

    This was originally generated by parsing and preprocessing the classic
    Reuters-21578 dataset, but the preprocessing code is no longer packaged
    with Keras. See this
    [GitHub discussion](https://github.com/keras-team/keras/issues/12072)
    for more info.

    Each newswire is encoded as a list of word indexes (integers).
    For convenience, words are indexed by overall frequency in the dataset,
    so that for instance the integer "3" encodes the 3rd most frequent word in
    the data. This allows for quick filtering operations such as:
    "only consider the top 10,000 most
    common words, but eliminate the top 20 most common words".

    As a convention, "0" does not stand for a specific word, but instead is used
    to encode any unknown word.

    Args:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: integer or None. Words are
          ranked by how often they occur (in the training set) and only
          the `num_words` most frequent words are kept. Any less frequent word
          will appear as `oov_char` value in the sequence data. If None,
          all words are kept. Defaults to `None`.
      skip_top: skip the top N most frequently occurring words
          (which may not be informative). These words will appear as
          `oov_char` value in the dataset. 0 means no words are
          skipped. Defaults to `0`.
      maxlen: int or None. Maximum sequence length.
          Any longer sequence will be truncated. None means no truncation.
          Defaults to `None`.
      test_split: Float between `0.` and `1.`. Fraction of the dataset to be
        used as test data. `0.2` means that 20% of the dataset is used as
        test data. Defaults to `0.2`.
      seed: int. Seed for reproducible data shuffling.
      start_char: int. The start of a sequence will be marked with this
          character. 0 is usually the padding character. Defaults to `1`.
      oov_char: int. The out-of-vocabulary character.
          Words that were cut out because of the `num_words` or
          `skip_top` limits will be replaced with this character.
      index_from: int. Index actual words with this index and higher.
      **kwargs: Used for backwards compatibility.

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train, x_test**: lists of sequences, which are lists of indexes
      (integers). If the num_words argument was specific, the maximum
      possible index value is `num_words - 1`. If the `maxlen` argument was
      specified, the largest possible sequence length is `maxlen`.

    **y_train, y_test**: lists of integer labels (1 or 0).

    Note: The 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """

    return reuters.load_data(test_split=test_split)