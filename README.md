# Temporal Graph MLP-Mixer

This repository contains the implementation of the Temporal Graph MLP-Mixer, an extension of the original Graph MLP-Mixer model designed to handle graph-structured data with temporal dynamics. This model aims to effectively capture both spatial and temporal dependencies within data that evolves over time, such as dynamic social networks, traffic networks, or financial transaction graphs.

## Overview

The Temporal Graph MLP-Mixer extends the Graph ViT/MLP-Mixer architecture by incorporating time-evolving features into the graph neural network framework. This allows the model to not only consider the structural and feature changes within the graph but also to model the evolution of these changes over time.

## Features

- **Temporal Patch Extraction**: Extends the METIS-based patch extraction to consider temporal slices of data, maintaining the connectivity information across time.
- **Temporal Positional Encoding**: Introduces temporal aspects to positional encodings to handle changes over successive time frames.
- **Dynamic Mixer Layers**: Mixer layers that adapt to temporal dynamics, allowing the model to integrate information across different time steps effectively.
- **Temporal Attention Mechanisms**: Incorporates a temporal version of the graph-based multi-head attention (gMHA) to focus on relevant temporal features.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- NetworkX
- METIS for Python

### Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/temporal-graph-mlp-mixer.git
cd temporal-graph-mlp-mixer
