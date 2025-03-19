import os
import time

def runtime():
    timestamp = time.time()  
    local_time = time.localtime(timestamp)  
    year, month, day, hour, minute, second, weekday, yearday, isdst = local_time    
    result = f"{year}_{month}_{day}_{hour}_{minute}_{second}"
    return result

def mkdir(path):
    name = runtime()
    path = f'{path}\\{name}'
    os.makedirs(path + '\\src', exist_ok=True)  # 使用os.makedirs代替os.popen和mkdir
    os.system(f'copy .\\*.py {path}\\src')  # 使用os.system和copy命令复制文件
    return path
