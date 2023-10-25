def list_of_dicts__to__dict_of_lists(lst: list | tuple):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    ```
    """
    assert isinstance(lst, (list, tuple)), type(lst)
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = dict()
    for d in lst:
        assert set(d.keys()) == set(keys), (d.keys(), keys)
        for k in keys:
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(d[k])
    return output_dict
