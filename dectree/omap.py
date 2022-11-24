from collections import OrderedDict
from typing import List, Union, Any, Dict


def to_omap(obj: Union[List, Dict, Any],
            recursive=False) -> Union[OrderedDict, List[OrderedDict], Any]:
    if not obj:
        return obj

    if _is_list_of_one_key_dicts(obj):
        dict_copy = OrderedDict()
        for item in obj:
            key, item = dict(item).popitem()
            dict_copy[key] = to_omap(item) if recursive else item
        return dict_copy

    if recursive:
        if isinstance(obj, list):
            list_copy = []
            for item in obj:
                list_copy.append(to_omap(item, recursive=True))
            return list_copy
        if isinstance(obj, dict):
            dict_copy = OrderedDict()
            for key, item in obj.items():
                dict_copy[key] = to_omap(item, recursive=True)
            return dict_copy

    return obj


def _is_list_of_one_key_dicts(l) -> bool:
    try:
        for item in l:
            # noinspection PyUnusedLocal
            (k, v), = item.items()
    except (AttributeError, TypeError):
        return False
    return True
