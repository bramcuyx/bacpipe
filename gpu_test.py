try:
    import torch

    # Check if CUDA is available
    print('Torch output: Is CUDA available', torch.cuda.is_available())

    # Check if CUDA is available
except:
    print('Torch not found')



try:
    from tensorflow.python.client import device_lib

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    print('Tensorflow GPU device:',get_available_gpus())

    print('All available devices according to tensorflow:',device_lib.list_local_devices())

except:
    print('Tensorflow not found')

try: 
    import subprocess
    print(subprocess.run(["nvidia-smi"]))

except:
    print('nvidia-smi not found')
