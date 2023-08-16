import re
from typing import List, Tuple, Optional, cast
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError
from rclip import utils
import torch
import torch.nn
from transformers import CLIPModel, CLIPProcessor
# import intel_extension_for_pytorch as ipex

QUERY_WITH_MULTIPLIER_RE = re.compile(r'^(?P<multiplier>(\d+(\.\d+)?|\.\d+|\d+\.)):(?P<query>.+)$')
QueryWithMultiplier = Tuple[float, str]


class Model:
  VECTOR_SIZE = 512
  _device = 'cpu'
  # declip - not out
  # fiber - not out
  # openclip laion
  # flava - ok trrying
  # _model_name = 'hakurei/waifu-diffusion'

  def __init__(self):
    model = cast(CLIPModel, CLIPModel.from_pretrained(self._model_name)).to(self._device)
    preprocess = CLIPProcessor.from_pretrained(self._model_name)
    # images = []
    # for _ in range(1):
    #   images.append(Image.new("RGB", (224, 224), "#FFFFFF"))
    #   images.append(Image.new("RGB", (224, 224)))
    #   images.append(Image.new("RGB", (1000, 1000), "#FF0000"))
    # images_processed = preprocess(images=images, return_tensors="pt")

    # ipex.enable_onednn_fusion(True)
    # with torch.no_grad():
    #   # model.image_model = ipex.optimize(model.image_model, conv_bn_folding=True, replace_dropout_with_identity=True, auto_kernel_selection=True)
    #   model.vision_model = torch.jit.trace(model.vision_model, images_processed.to(self._device)['pixel_values'], strict=False, check_trace=True)
    #   model.vision_model = torch.jit.freeze(model.vision_model)

    self._model = model
    self._preprocess = preprocess

  # def get_image_features(self, images):
  #   image_outputs = self._model.vision_model(
  #       pixel_values=images['pixel_values'],
  #   )

  #   pooled_output = image_outputs[1]  # last_hidden_state
  #   image_features = self._model.visual_projection(pooled_output)
  #   return image_features

  def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
    images_preprocessed = self._preprocess(images=images, return_tensors="pt")

    with torch.no_grad():
      image_features = self._model.get_image_features(**images_preprocessed)
      image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

  def compute_text_features(self, text: List[str]) -> np.ndarray:
    import torch
    with torch.no_grad():
      text_encoded = self._model.get_text_features(**self._preprocess(text, return_tensors="pt"))
      text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

  @staticmethod
  def _extract_query_multiplier(query: str) -> QueryWithMultiplier:
    match = QUERY_WITH_MULTIPLIER_RE.match(query)
    if not match:
      return 1., query
    multiplier = float(match.group('multiplier'))
    query = match.group('query')
    return multiplier, query

  @staticmethod
  def _group_queries_by_type(queries: List[str]) -> Tuple[
    List[QueryWithMultiplier],
    List[QueryWithMultiplier],
    List[QueryWithMultiplier]
  ]:
    phrase_queries: List[Tuple[float, str]] = []
    local_file_queries: List[Tuple[float, str]] = []
    url_queries: List[Tuple[float, str]] = []
    for query in queries:
        multiplier, query = Model._extract_query_multiplier(query)
        if utils.is_http_url(query):
          url_queries.append((multiplier, query))
        elif utils.is_file_path(query):
          local_file_queries.append((multiplier, query))
        else:
          phrase_queries.append((multiplier, query))
    return phrase_queries, local_file_queries, url_queries

  def compute_features_for_queries(self, queries: List[str]) -> np.ndarray:
    text_features: Optional[np.ndarray] = None
    image_features: Optional[np.ndarray] = None
    phrases, files, urls = self._group_queries_by_type(queries)
    if phrases:
      phrase_multipliers, phrase_queries = cast(Tuple[Tuple[float], Tuple[str]], zip(*phrases))
      phrase_multipliers_np = np.array(phrase_multipliers).reshape(-1, 1)
      text_features = np.add.reduce(self.compute_text_features([*phrase_queries]) * phrase_multipliers_np)
    if files or urls:
      file_multipliers, file_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(files))) if files else ((), ())
      url_multipliers, url_paths = cast(Tuple[Tuple[float], Tuple[str]], zip(*(urls))) if urls else ((), ())
      try:
        images = ([utils.download_image(q) for q in url_paths] +
                  [utils.read_image(q) for q in file_paths])
      except FileNotFoundError as e:
        print(f'File "{e.filename}" not found. Check if you have typos in the filename.')
        sys.exit(1)
      except UnidentifiedImageError as e:
        print(f'File "{e.filename}" is not an image. You can only use image files or text as queries.')
        sys.exit(1)
      image_multipliers = np.array(url_multipliers + file_multipliers)
      image_features = np.add.reduce(self.compute_image_features(images) * image_multipliers.reshape(-1, 1))

    if text_features is not None and image_features is not None:
        return text_features + image_features
    elif text_features is not None:
        return text_features
    elif image_features is not None:
        return image_features
    else:
        return np.zeros(Model.VECTOR_SIZE)

  def compute_similarities_to_text(
      self, item_features: np.ndarray,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    positive_features = self.compute_features_for_queries(positive_queries)
    negative_features = self.compute_features_for_queries(negative_queries)

    features = positive_features - negative_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities

class CLIP(Model):
  _model_name = 'openai/clip-vit-base-patch32'

class OpenCLIP(Model):
  _model_name = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'

model_dict = {
  'clip': CLIP,
  'openclip': OpenCLIP,
}

default_model = 'openclip'
