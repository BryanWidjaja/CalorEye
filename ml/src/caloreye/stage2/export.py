from pathlib import Path

import timm
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from caloreye.export.spike import collect_delegation_report

MODEL_NAME = "efficientnet_b0.ra4_e3600_r224_in1k"
CLASS_COUNT = 7
INPUT_SIZE = 224

def load_trained_classifier(weights_path: Path) -> torch.nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=CLASS_COUNT)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    return model.eval()

def export_trained_classifier(weights_path: Path, program_path: Path) -> dict:
    classifier = load_trained_classifier(weights_path)
    example_input = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).contiguous()
    exported = torch.export.export(classifier, (example_input,))
    edge_program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    program_path.write_bytes(edge_program.to_executorch().buffer)

    return collect_delegation_report(edge_program)
