#!/usr/bin/env python3

# lix19937

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, metavar="ONNX_MODEL", help="ONNX model to check.")
args = parser.parse_args()

import onnx
model = onnx.load(args.model)
onnx.checker.check_model(model)

