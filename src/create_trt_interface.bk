import tensorflow.contrib.tensorrt as trt

# trt.create_inference_graph(input_saved_model_dir='/home/pawan/20180408-102900', output_saved_model_dir='/home/pawan/20180408-102900/trt')

converted_graph_def = trt.create_inference_graph(input_graph_def='/home/pawan/20180408-102900/frozen_graph.pb',
                                                 outputs=['logits', 'classes'])
