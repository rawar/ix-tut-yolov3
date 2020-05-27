# Beispielcode für NVIDIAs Jetson Nano und Xavier NX

## Keras Modelle nach ONNX konvertieren

Modelle, welche mit Hilfe von tf.keras traniert worden sind, können auch nativ auf den NVIDIA Jetsons ausgeführt werden. Allerdings nuzt Tensorflow trotz
GPU-Unterstützung, nicht das voll Leistungsspektrum von NVIDIAs Hardware aus. Das führt leider dazu, dass die Verarbeitungsgeschwindigkeit bei der Inference
eher bescheiden ist. Besonders wenn man Echtzeitbild- oder -Objektekennung implementierten möchte, kommt man an einer etwas hardware-naheren Programmierung
der Jetsons nicht drum herum. Um dies mit Hilfe des TensorRT-Frameworks, welches für alle Mitglieder der Jetson-Famlie verfügbar ist, zu realsieren, müssen die
Keras-Modelle erstmal in ein Zwischenformat gebracht werden, da TensorRT die Keras-Modelle nicht einfach einlesen kann. Das Format der Wahl ist 
hiebei das ONNX Format. Passende Konverter sind mit Hilfe von

```
$ pip install git+https://github.com/microsoft/onnxconverter-common
$ pip install git+https://github.com/onnx/keras-onnx
```

schnell installiert.


## ONNX Modelle in TensorRT konvertieren

Ausgangspunkt für die maschinennahe Ausführung von Deep Learning Modellen auf den Edge-Systemen der Jetson-Familie ist das TensorRT-Framework.
TensorRT stellt dabei einen Modell-Compiler und eine Modell-Runtime-Umgebung zur Verfügung. Diese auch als ```Engine```bezeichnete Laufzeitumgebung
ermöglicht die Ausführung für die spezifische (GPU-) Hardware optimierte Modelle.

Mit Hilfe des Python3-Scripts ```onnx_to_engine.py``` können beliebige Modelle im ONNX-Format in eine TensorRT-Engine konvertiert werden.

## Modelle nutzen mit TensorRT

Mit TensorRT lassen sich Modelle auch in unterschiedliche Genauigkeiten überführen. Neben FP32- werden auch FP16- und INT8-Genauigkeiten unterstützt.

