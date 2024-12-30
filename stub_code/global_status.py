import json

# # 创建一个简单的 JSON 对象
# data = {
#     "name": "Alice",
#     "age": 30,
#     "city": "New York"
# }

# # 将 Python 字典转换为 JSON 对象
# json_data = json.dumps(data)

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class GlobalStatus:
    def __init__(self):
        self._jsonobj = \
            {
                "runningStatus" : "就绪",
                "modelFile" : "none",
                "operatorOutputDir" : "none",
                "platform" : "dcu",
                "optimizeOperators" : ["mm","pool","conv2d"],
                "tuningConfigFile" : "none"
            }
        self._jsonstr = json.dumps(self._jsonobj)
    def set_runningStatus(self,val:str) :
        self._jsonobj["runningStatus"] = val
    def set_modelFile(self,val:str) :
        self._jsonobj["modelFile"] = val
    def set_operatorOutputDir(self,val:str) :
        self._jsonobj["operatorOutputDir"] = val
    def set_platform(self,val:str) :
        self._jsonobj["platform"] = val
    def set_optimizeOperators(self,val:str) :
        self._jsonobj["optimizeOperators"] = val
    def set_tuningConfigFile(self,val:str) :
        self._jsonobj["tuningConfigFile"] = val
    def get(self,key : str) :
        return self._jsonobj[key]
    def to_string(self) -> str :
        self._jsonstr = json.dumps(self._jsonobj)
        return self._jsonstr

