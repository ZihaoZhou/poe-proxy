from typing import Union, List, Dict
from collections import defaultdict


def delete_index_from_message(data):
    if isinstance(data, dict):
        if "message" in data:
            delete_index(data["message"])
        else:
            for value in data.values():
                delete_index_from_message(value)
    elif isinstance(data, list):
        for item in data:
            delete_index_from_message(item)


def delete_index(obj):
    if isinstance(obj, dict):
        obj.pop("index", None)
        for value in obj.values():
            delete_index(value)
    elif isinstance(obj, list):
        for item in obj:
            delete_index(item)


def aggregate_json(data: Union[List, Dict]) -> Union[List, Dict]:
    if isinstance(data, list):
        return [aggregate_json(item) for item in data]
    elif isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "delta":
                result["message"] = aggregate_json(value)
            elif isinstance(value, (dict, list)):
                result[key] = aggregate_json(value)
            else:
                result[key] = value
        return result
    else:
        return data


def aggregate_chunk(json_entries: List[Dict]) -> Dict:
    aggregated = aggregate_json(json_entries)
    data = merge_messages(aggregated)
    delete_index_from_message(data)
    # Add 'role': 'assistant' to the message if it doesn't exist
    for choice in data['choices']:
        if 'message' in choice:
            if 'role' not in choice['message']:
                choice['message']['role'] = 'assistant'
    
    # Remove '.chunk' from object type if it exists
    if 'object' in data and data['object'].endswith('.chunk'):
        data['object'] = data['object'].replace('.chunk', '')
    return data


def merge_messages(messages: List[Dict]) -> Dict:
    result = {}
    for message in messages:
        for key, value in message.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = merge_messages([result[key], value])
            elif isinstance(value, list) and isinstance(result[key], list):
                result[key] = merge_lists(result[key], value)
            elif value is not None:
                if result[key] is None:
                    result[key] = value
                elif value != result[key]:
                    if isinstance(value, str) and isinstance(result[key], str):
                        result[key] += value
                    else:
                        result[key] = [result[key], value]
    return result


def merge_lists(list1: List, list2: List) -> List:
    # Check if the lists contain dictionaries with an "index" key
    if all(isinstance(item, dict) and "index" in item for item in list1 + list2):
        return merge_indexed_lists(list1 + list2)
    
    # If lengths are different or items are not dictionaries, just concatenate
    if len(list1) != len(list2) or not all(isinstance(item, dict) for item in list1 + list2):
        return list1 + list2
    
    return [merge_messages([item1, item2]) for item1, item2 in zip(list1, list2)]


def merge_indexed_lists(items: List[Dict]) -> List[Dict]:
    indexed_data = defaultdict(list)
    for item in items:
        indexed_data[item["index"]].append(item)
    
    result = []
    for index in sorted(indexed_data.keys()):
        merged_item = merge_messages(indexed_data[index])
        result.append(merged_item)
    return result
