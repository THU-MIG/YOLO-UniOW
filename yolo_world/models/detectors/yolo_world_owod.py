# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class OWODDetector(YOLODetector):
    """Implementation of Open-World YOLO"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes: int = 80,
                 num_test_classes: int = 80,
                 num_prev_classes: int = 0,
                 num_prompts: int = 80,
                 prompt_dim: int = 512,
                 embedding_path: str = '',
                 unknown_embedding_path: str = '',
                 anchor_embedding_path: str = '',
                 embedding_mask: Union[List, int] = None,
                 freeze_prompt: bool = False,
                 use_mlp_adapter: bool = False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.num_prev_classes = num_prev_classes
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if len(embedding_path) > 0:
            self.embeddings = torch.nn.Parameter(
                torch.from_numpy(np.load(embedding_path)).float())
        else:
            # random init
            embeddings = nn.functional.normalize(torch.randn(
                (num_train_classes, prompt_dim)),
                                                    dim=-1)
            self.embeddings = nn.Parameter(embeddings)

        if len(unknown_embedding_path) > 0:
            unknown_embeddings = nn.Parameter(torch.from_numpy(
                np.load(unknown_embedding_path)).float())
            self.embeddings = nn.Parameter(torch.cat([self.embeddings, unknown_embeddings], dim=0))
        
        if len(anchor_embedding_path) > 0:
            anchor_embeddings = nn.Parameter(torch.from_numpy(
                np.load(anchor_embedding_path)).float())
            self.embeddings = nn.Parameter(torch.cat([self.embeddings, anchor_embeddings], dim=0))

        if self.freeze_prompt:
            self.embeddings.requires_grad = False
        else:
            self.embeddings.requires_grad = True

        if embedding_mask:
            if isinstance(embedding_mask, int):
                self._grad_mask = torch.ones(num_train_classes, dtype=torch.bool)[:, None]
                self._grad_mask[:embedding_mask] = False
            else:
                self._grad_mask = torch.Tensor(embedding_mask).bool()[:, None]
            assert len(self._grad_mask) == num_train_classes
            self.embeddings.register_hook(lambda grad: grad * self._grad_mask.to(grad.device))

        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                nn.Linear(prompt_dim * 2, prompt_dim))
        else:
            self.adapter = None
                
    def update_embeddings(self, embeddings):
        # update embeddings when loading from checkpoint
        prev_embeddings = embeddings[:self.num_prev_classes]
        cur_embeddings = self.embeddings[self.num_prev_classes:].detach().cpu()
        embeddings = torch.cat([prev_embeddings, cur_embeddings], dim=0)
        return embeddings

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats,
                                        batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes

        results_list = self.bbox_head.predict(img_feats,
                                                txt_feats,
                                                batch_data_samples,
                                                rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        # use embeddings
        txt_feats = self.embeddings[None]
        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
