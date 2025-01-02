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
class UserInterface :
    def __init__(self): 
        # 当前工作目录，即程序所在路径
        self.cwd = ''
        # user inputs
        self.m_modelFilePath = ''
        self.m_platName = ''
        self.m_codegenPath = ''
        self.m_tempPath = ''
        self.m_tuningConfigFile = ''
        # json 对象 ,用于展示当前程序状态
        self._jsonobj = \
            {
                "cwd" : self.cwd,
                "runningStatus" : "就绪",   # 当前运行状态(out) （就绪、分析中、算子生成中。。。）
                "modelFile" : self.m_modelFilePath,   # 模型文件路径(user input  by文本框)
                "operatorOutputDir" : self.m_codegenPath,  # 算子输出路径(user input by文本框)
                "platform" : self.m_platName, # 当前平台 (user input, by下拉菜单选择) : dcu,mlu,npu
                "optimizeOperators" : ["mm","pool","conv2d"],   # 待优化算子(user input, by通过网页多选)
                "tuningConfigFile" : self.m_tuningConfigFile,    # 调优配置文件选择(user input, by通过预制的下拉列表从几个现成的confg里选择)
                "tempPath" : self.m_tempPath # 临时文件目录,用于输出一些中间文件 (user input,  by文本框)
            }
        self._jsonstr = json.dumps(self._jsonobj)   # json字符串，和 _jsonobj 对应
    
    def build_with_jsonstr(self,json_str : str) :
        self._jsonobj = json.loads(json_str)
        self._jsonstr = json_str
        self.m_modelFilePath = self._jsonobj['modelFile'] 
        self.m_codegenPath = self._jsonobj['operatorOutputDir'] 
        self.m_tempPath = self._jsonobj['tempPath'] 
        self.m_platName = self._jsonobj['platform'] 
        self.m_tuningConfigFile = self._jsonobj["tuningConfigFile"]
        self.cwd = self._jsonobj["cwd"]
        
    def set_modelFilePath(self, modelFilePath : str) :
        self.m_modelFilePath = modelFilePath
        self._jsonobj['modelFile'] = modelFilePath
        
    def set_codegenPath(self, codegenPath : str) :
        self.m_codegenPath = codegenPath
        self._jsonobj['operatorOutputDir'] = codegenPath
        
    def set_tempPath(self, tempPath : str) :
        self.m_tempPath = tempPath
        self._jsonobj['tempPath'] = tempPath
        
    def set_platform(self,platName : str) : 
        if platName == 'dcu' or platName == 'npu' or platName == 'mlu':
            print(f"current platform = {platName}")
            self.m_platName = platName
            self._jsonobj['platform'] = platName
        else:
            assert("invalid platform" and False)
        
    def set_runningStatus(self,val:str):
        self._jsonobj["runningStatus"] = val
        return self.dumpJsonString()
        
    def set_optimizeOperators(self,val:str):
        self._jsonobj["optimizeOperators"] = val
        
    def set_tuningConfigFile(self, opt:str):
        if opt != 'standard' and opt != 'simple' and opt != 'complex':
            assert(False and "invalid option!")
        val = self.tuneCfgFiles()[opt]
        self._jsonobj["tuningConfigFile"] = val
        self.m_tuningConfigFile = val
    
    def tuneCfgFiles(self) : 
        assert(len(self.cwd) > 0)
        obj= {
            "standard" : f"{self.cwd}/tune_configs_standard.yaml",
            "simple" : f"{self.cwd}/tune_configs_simple.yaml",
            "complex" : f"{self.cwd}/tune_configs_complex.yaml",
        }
        return obj
    
    def dumpJsonString(self) :
        self._jsonstr = json.dumps(self._jsonobj)   # json字符串，和 _jsonobj 对应
        return self._jsonstr    