# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The FNO construction below is adapted from
# examples/cfd/darcy_fno/train_fno_darcy.py of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by Adebanji Adelowo (2026):
# moved into a function, and the number of Fourier modes made an argument.

from omegaconf import DictConfig
from physicsnemo.models.fno import FNO


def build_fno(cfg: DictConfig, fno_modes: int | None = None) -> FNO:
    """Build the PhysicsNeMo FNO of the Darcy example.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with the upstream ``arch`` node.
    fno_modes : int, optional
        Number of retained Fourier modes per dimension. Defaults to ``cfg.arch.fno.fno_modes``.
    """
    modes = cfg.arch.fno.fno_modes if fno_modes is None else fno_modes
    return FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=modes,
        padding=cfg.arch.fno.padding,
    )


def max_fno_modes(resolution: int, padding: int) -> int:
    """Largest number of modes the spectral convolution can retain.

    The layer keeps ``modes`` coefficients of the real FFT of the padded grid, which has
    ``(resolution + padding) // 2 + 1`` coefficients in its last dimension.
    """
    return (resolution + padding) // 2 + 1


def count_parameters(model) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
