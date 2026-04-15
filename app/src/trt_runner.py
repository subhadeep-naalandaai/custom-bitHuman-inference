"""
src/trt_runner.py
Python replacement for trt_runner.cpython-310-x86_64-linux-gnu.so

TensorRT engine runner with per-session execution contexts.

Classes:
  TRTContext — per-session IExecutionContext + I/O buffer management
  TRTRunner  — loads a TRT engine and creates per-session TRTContext objects

When TensorRT is not installed the classes still import and instantiate, but
infer() will raise RuntimeError.  The rest of the system (trt_pipeline.py)
gracefully falls back to PyTorch in that case.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("expression-avatar.trt_runner")


# ---------------------------------------------------------------------------
# Per-session execution context
# ---------------------------------------------------------------------------

class TRTContext:
    """Per-session TRT execution context."""

    def __init__(self, engine):
        """
        Create a new per-session execution context.

        Args:
            engine: tensorrt.ICudaEngine (already deserialized by TRTRunner)
        """
        self._engine  = engine
        self._context = engine.create_execution_context()

        # Classify tensors as inputs / outputs
        self._input_names:  list = []
        self._output_names: list = []

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name).name == "INPUT":
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def infer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run a synchronous forward pass through the TRT engine.

        Args:
            inputs: mapping of tensor-name → CUDA torch.Tensor

        Returns:
            mapping of tensor-name → CUDA torch.Tensor for each output
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError("TensorRT is not installed — cannot run TRT inference")

        outputs: Dict[str, torch.Tensor] = {}

        # Set dynamic shapes and register input pointers
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            self._context.set_input_shape(name, tuple(tensor.shape))
            self._context.set_tensor_address(name, tensor.data_ptr())

        # Allocate output buffers and register their pointers
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            dtype_str = str(self._engine.get_tensor_dtype(name))
            torch_dtype = torch.float16 if "HALF" in dtype_str else torch.float32
            out = torch.empty(shape, dtype=torch_dtype, device="cuda")
            self._context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

        # Execute asynchronously on the current CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        self._context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        return outputs


# ---------------------------------------------------------------------------
# Engine loader
# ---------------------------------------------------------------------------

class TRTRunner:
    """TensorRT engine loader — creates per-session TRTContext objects."""

    def __init__(self, engine_path: str):
        """
        Load a serialised TRT engine from disk.

        Args:
            engine_path: path to *.trt engine file
        """
        self._engine_path = str(engine_path)
        self._engine      = None

        try:
            import tensorrt as trt
        except ImportError:
            logger.warning("[TRTRunner] TensorRT not installed — engine load skipped")
            return

        logger.info(f"[TRTRunner] Loading engine: {self._engine_path}")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(self._engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"[TRTRunner] Failed to deserialise engine: {self._engine_path}")

        logger.info(f"[TRTRunner] Engine loaded — {self._engine.num_io_tensors} tensors")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_context(self) -> TRTContext:
        """Create a new per-session execution context."""
        if self._engine is None:
            raise RuntimeError("[TRTRunner] Engine not loaded")
        return TRTContext(self._engine)

    def get_engine_info(self) -> dict:
        """Return a dict of tensor-name → {mode, dtype} for all I/O tensors."""
        if self._engine is None:
            return {}
        info = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            info[name] = {
                "mode":  self._engine.get_tensor_mode(name).name,
                "dtype": str(self._engine.get_tensor_dtype(name)),
            }
        return info

    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
        context: Optional[TRTContext] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference using the given context (or default).

        Args:
            inputs:  tensor-name → CUDA tensor
            context: per-session TRTContext; a temporary one is created if None

        Returns:
            tensor-name → CUDA output tensor
        """
        if self._engine is None:
            raise RuntimeError("[TRTRunner] Engine not loaded")
        ctx = context if context is not None else TRTContext(self._engine)
        return ctx.infer(inputs)
