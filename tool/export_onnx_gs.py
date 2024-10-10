import onnx_graphsurgeon as gs
import numpy as np
import onnx

def CreateModel():
  input_ = gs.Variable(name="input", dtype=np.float32, shape=(1, 256, 60, 36))
  output_ = gs.Variable(name="output", dtype=np.float32, shape=(1, 256//4, 60*2, 36*2))
  node = gs.Node(op="PixelShufflePlugin", attrs={"upscaleFactor": 2}, inputs=[input_], outputs=[output_])

  graph = gs.Graph(nodes=[node], inputs=[input_], outputs=[output_])
  onnx.save(gs.export_onnx(graph), "pixelShuffle.onnx")
  # import netron
  # netron.start("pixelShuffle.onnx")

def main():
  CreateModel()

if __name__ == '__main__':
  main()

#
# trtexec --onnx=./pixelShuffle.onnx  --plugins=./libnni_plugin_custom.so --verbose
#
