import json

from dataclass_wizard import JSONSerializable as _JSONSerializable


class JSONSerializable(_JSONSerializable):
    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as fp:
            data = json.load(fp)
        return cls.from_dict(data)
