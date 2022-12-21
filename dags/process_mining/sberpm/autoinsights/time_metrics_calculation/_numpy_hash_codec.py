from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Iterable, Sequence

from numpy import array, empty, frombuffer, int64


@dataclass
class ArrayToHashSequenceCodec:
    _dtype: type
    _bytes2hash_int_collection: DefaultDict = field(default_factory=lambda: defaultdict(bytes))

    @property
    def get_array_hashes(self) -> Iterable[int64]:
        self.__check_arrays_encoded()

        return array(tuple(self._bytes2hash_int_collection))

    @property
    def get_arrays(self) -> Sequence[array]:
        self.__check_arrays_encoded()

        _numpy_array_collection = empty(self.get_array_hashes.shape, dtype="object")
        self.__decode_arrays(_numpy_array_collection)

        return _numpy_array_collection

    def encode_arrays(self, numpy_array_collection: Iterable[array]) -> None:
        array_bytes_collection = map(lambda arr: arr.tobytes(), numpy_array_collection)

        for array_bytes in array_bytes_collection:
            hashed_bytes = (
                hash(array_bytes) + 11
                if array_bytes in self._bytes2hash_int_collection.values()
                else hash(array_bytes)
            )

            self._bytes2hash_int_collection[hashed_bytes] = array_bytes

    def __decode_arrays(self, collection_shape_array: Sequence[array]) -> None:
        for pos, array_hash in enumerate(self._bytes2hash_int_collection):
            collection_shape_array[pos] = frombuffer(
                self._bytes2hash_int_collection[array_hash], dtype=self._dtype
            )

    def __check_arrays_encoded(self) -> None:
        if not self._bytes2hash_int_collection:
            raise RuntimeError("You must encode arrays collection first")
