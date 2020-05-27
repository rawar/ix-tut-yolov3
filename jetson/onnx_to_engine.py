from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import utils
import os
import common

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def get_engine(onnx_file_path, engine_file_path=""):

    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            if not os.path.exists(onnx_file_path):
                print('ONNX Datei {} nicht gefunden.'.format(onnx_file_path))
                exit(0)
            print('Lade ONNX Datei von {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Parse die ONNX Datei {}'.format(onnx_file_path))
                if not parser.parse(model.read()):
                    print ('ERROR: Fehler in ONNX Datei.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
           
            print(f"Network Name {network.name}")
            print(f"Anzahl Rückgaben {network.num_outputs}")
            print(f"Anzahl Layer {network.num_layers}")
            print(f"Eingabe Layer Form {network.get_input(0).shape}")
            print(f"Ausgabe Layer Form {network.get_output(0).shape}")

            network.get_input(0).shape = (1, 416, 416, 3)

            print('ONNX Dateiparsing beendet')
            print('Erzeuge TensorRT Engine aus der Datei {}; Dies kann einige Minuten dauern...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("TensorRT Engine Erzeugung beendet")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Lese bestehende TensorRT Engine Datei aus {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    onnx_file_path = 'models/starwars_yolov3.onnx'
    engine_file_path = 'models/starwars_yolov3.trt'
    trt_engine = get_engine(onnx_file_path, engine_file_path)

if __name__ == '__main__':
    main()
