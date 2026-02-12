from typing import Any, Optional, Union

from pathlib import Path
import io
import lmdb
import pickle
import gzip
import bz2
import lzma

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _default_encode(data: Any, protocol: int) -> bytes:
    return pickle.dumps(data, protocol=protocol)


def _ascii_encode(data: str) -> bytes:
    return data.encode("ascii")


def _default_decode(data: bytes) -> Any:
    return pickle.loads(data)


def _default_decompress(data: bytes) -> bytes:
    return data


def _decompress(compression: Optional[str]):
    if compression is None:
        _decompress = _default_decompress
    elif compression == "gzip":
        _decompress = gzip.decompress
    elif compression == "bz2":
        _decompress = bz2.decompress
    elif compression == "lzma":
        _decompress = lzma.decompress
    else:
        raise ValueError(f"Unknown compression algorithm: {compression}")

    return _decompress


class Database(object):
    _database = None
    _protocol = None
    _length = None

    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = False,
        pre_open: bool = False,
        compression: Optional[str] = None,
    ):
        """
        Base class for LMDB-backed databases.

        :param path: Path to the database.
        :param readahead: Enables the filesystem readahead mechanism.
        :param pre_open: If set to True, the first iterations will be faster, but it will raise error when doing multi-gpu training. If set to False, the database will open when you will retrieve the first item.
        """
        if not isinstance(path, str):
            path = str(path)

        self.path = path
        self.readahead = readahead
        self.pre_open = pre_open
        self._decompress = _decompress(compression)

        self._has_fetched_an_item = False

    @property
    def database(self):
        if self._database is None:
            self._database = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )
        return self._database

    @database.deleter
    def database(self):
        if self._database is not None:
            self._database.close()
            self._database = None

    @property
    def protocol(self):
        """
        Read the pickle protocol contained in the database.

        :return: The set of available keys.
        """
        if self._protocol is None:
            self._protocol = self._get(
                item="protocol",
                encode_key=_ascii_encode,
                decompress_value=_default_decompress,
                decode_value=_default_decode,
            )
        return self._protocol

    @property
    def keys(self):
        """
        Read the keys contained in the database.

        :return: The set of available keys.
        """
        protocol = self.protocol
        keys = self._get(
            item="keys",
            encode_key=lambda key: _default_encode(key, protocol=protocol),
            decompress_value=_default_decompress,
            decode_value=_default_decode,
        )
        return keys

    def __len__(self):
        """
        Returns the number of keys available in the database.

        :return: The number of keys.
        """
        if self._length is None:
            self._length = len(self.keys)
        return self._length

    def __getitem__(self, item):
        """
        Retrieves an item or a list of items from the database.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._has_fetched_an_item = True
        if not isinstance(item, list):
            item = self._get(
                item=item,
                encode_key=self._encode_key,
                decompress_value=self._decompress_value,
                decode_value=self._decode_value,
            )
        else:
            item = self._gets(
                items=item,
                encode_keys=self._encode_keys,
                decompress_values=self._decompress_values,
                decode_values=self._decode_values,
            )
        return item

    def _get(self, item, encode_key, decompress_value, decode_value):
        """
        Instantiates a transaction and its associated cursor to fetch an item.

        :param item: A key.
        :param encode_key:
        :param decode_value:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                item = self._fetch(
                    cursor=cursor,
                    key=item,
                    encode_key=encode_key,
                    decompress_value=decompress_value,
                    decode_value=decode_value,
                )
        self._keep_database()
        return item

    def _gets(self, items, encode_keys, decompress_values, decode_values):
        """
        Instantiates a transaction and its associated cursor to fetch a list of items.

        :param items: A list of keys.
        :param encode_keys:
        :param decode_values:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                items = self._fetchs(
                    cursor=cursor,
                    keys=items,
                    encode_keys=encode_keys,
                    decompress_values=decompress_values,
                    decode_values=decode_values,
                )
        self._keep_database()
        return items

    def _fetch(self, cursor, key, encode_key, decompress_value, decode_value):
        """
        Retrieve a value given a key.

        :param cursor:
        :param key: A key.
        :param encode_key:
        :param decode_value:
        :return: A value.
        """
        key = encode_key(key)
        value = cursor.get(key)
        value = decompress_value(value)
        value = decode_value(value)
        return value

    def _fetchs(self, cursor, keys, encode_keys, decompress_values, decode_values):
        """
        Retrieve a list of values given a list of keys.

        :param cursor:
        :param keys: A list of keys.
        :param encode_keys:
        :param decode_values:
        :return: A list of values.
        """
        keys = encode_keys(keys)
        _, values = list(zip(*cursor.getmulti(keys)))
        values = decompress_values(values)
        values = decode_values(values)
        return values

    def _encode_key(self, key: Any) -> bytes:
        """
        Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return pickle.dumps(key, protocol=self.protocol)

    def _encode_keys(self, keys):
        """
        Converts keys into byte keys.

        :param keys: A list of keys.
        :return: A list of byte keys.
        """
        return [self._encode_key(key=key) for key in keys]

    def _decompress_value(self, value: bytes) -> bytes:
        return self._decompress(value)

    def _decompress_values(self, values) :
        return [self._decompress_value(value=value) for value in values]

    def _decode_value(self, value: bytes) -> Any:
        """
        Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return pickle.loads(value)

    def _decode_values(self, values):
        """
        Converts bytes values back into values.

        :param values: A list of byte values.
        :return: A list of values.
        """
        return [self._decode_value(value=value) for value in values]

    def _keep_database(self):
        """
        Checks if the database must be deleted.

        :return:
        """
        if not self.pre_open and not self._has_fetched_an_item:
            del self.database

    def __iter__(self):
        """
        Provides an iterator over the keys when iterating over the database.

        :return: An iterator on the keys.
        """
        return iter(self.keys)

    def __del__(self):
        """
        Closes the database properly.
        """
        del self.database


class ImageDatabase(Database):
    def _decode_value(self, value: bytes):
        """
        Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value))


class MaskDatabase(ImageDatabase):
    def _decode_value(self, value: bytes):
        """
        Converts a byte image back into a PIL Image.

        :param value: A byte image.
        :return: A PIL Image image.
        """
        return Image.open(io.BytesIO(value)).convert("1")


class LabelDatabase(Database):
    pass


class ArrayDatabase(Database):
    _dtype = None
    _shape = None

    @property
    def dtype(self):
        if self._dtype is None:
            protocol = self.protocol
            self._dtype = self._get(
                item="dtype",
                encode_key=lambda key: _default_encode(key, protocol=protocol),
                decompress_value=_default_decompress,
                decode_value=_default_decode,
            )
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            protocol = self.protocol
            self._shape = self._get(
                item="shape",
                encode_key=lambda key: _default_encode(key, protocol=protocol),
                decompress_value=_default_decompress,
                decode_value=_default_decode,
            )
        return self._shape

    def _decode_value(self, value: bytes) -> ndarray:
        return np.frombuffer(value, dtype=self.dtype).reshape(self.shape)

    def _decode_values(self, values):
        shape = (len(values),) + self.shape
        return np.frombuffer(b"".join(values), dtype=self.dtype).reshape(shape)


class TensorDatabase(ArrayDatabase):
    def _decode_value(self, value: bytes) -> Tensor:
        return torch.from_numpy(super(TensorDatabase, self)._decode_value(value))

    def _decode_values(self, values):
        return torch.from_numpy(super(TensorDatabase, self)._decode_values(values))
