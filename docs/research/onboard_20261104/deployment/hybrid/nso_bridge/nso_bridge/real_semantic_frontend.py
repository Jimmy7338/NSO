"""ROS-free, pinned YOLOE text-prompt instance segmentation on measured RGB.

Prompt labels and uncalibrated model confidence are predictions, not verified
object identities. Unknown pixels remain 0. Boxes never supply surface masks.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

DEFAULT_PROMPTS = ('shelves', 'bookcase', 'closed equipment cabinet', 'cabinet',
                   'table', 'chair', 'display counter', 'floor', 'wall')
V10_MAPPING = {'shelves': 2, 'bookcase': 2, 'closed equipment cabinet': 3}
PRIOR_PROFILES = {
    'strict_v10': dict(V10_MAPPING),
    'real_scene_structural_v1': dict(V10_MAPPING, cabinet=3, **{'display counter': 3}),
}


def prior_interpretations(profile):
    if profile not in PRIOR_PROFILES:
        raise ValueError('unsupported semantic prior profile: ' + str(profile))
    descriptions = {name: 'no V10 structural prior; retain original prompt namespace' for name in DEFAULT_PROMPTS}
    descriptions.update(shelves='original V10 class2 open multilayer structural prior',
        bookcase='original V10 class2 open multilayer structural prior',
        **{'closed equipment cabinet': 'original V10 class3 prior from predicted closed equipment cabinet category'})
    if profile == 'real_scene_structural_v1':
        for name in ('cabinet', 'display counter'):
            descriptions[name] = ('explicit real-scene extension: assume closed enclosure structural prior (V10 slot3); '
                'source category remains ' + name + '; not recognition of a closed equipment cabinet or measurement of hidden structure')
    return descriptions


MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'real_semantics_20261030'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class RealSemanticFrontend:
    def __init__(self, model_dir=MODEL_DIR, prompts=DEFAULT_PROMPTS, confidence=.25, imgsz=640, prior_profile='strict_v10'):
        self.prior_profile = prior_profile
        self.interpretations = prior_interpretations(prior_profile)
        self.mapping = dict(PRIOR_PROFILES[prior_profile])
        import torch
        from ultralytics import YOLOE
        import ultralytics
        if ultralytics.__version__ != '8.3.200':
            raise RuntimeError('pinned Ultralytics 8.3.200 required')
        self.model_dir = Path(model_dir)
        self.prompts = tuple(prompts)
        if self.prompts != DEFAULT_PROMPTS:
            raise ValueError('prompt changes require a separately generated embedding manifest')
        self.manifest = json.loads((self.model_dir / 'MODEL_MANIFEST.json').read_text())
        if tuple(self.manifest['prompts']) != self.prompts:
            raise ValueError('embedding prompt manifest mismatch')
        for row in self.manifest['files']:
            if file_sha256(self.model_dir / row['filename']) != row['sha256']:
                raise ValueError('pinned model/embedding SHA256 mismatch: ' + row['filename'])
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        from ultralytics.utils import torch_utils
        torch_utils.NUM_THREADS = 1
        torch.set_num_threads(1)
        self.model = YOLOE(str(self.model_dir / 'yoloe-v8s-seg.pt'))
        self.model.to('cpu')
        embeddings = torch.load(self.model_dir / 'prompt_embeddings.pt', map_location='cpu', weights_only=True)
        self.model.set_classes(list(self.prompts), embeddings)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)

    def predict(self, rgb, validdepth, *, frame_id=None):
        """Return semantic HxW, receipt and instances with actual boolean masks.

        validdepth may be a boolean HxW validity mask or measured depth in m.
        Caller retains the RGB/depth calibration and observation association.
        """
        rgb = np.asarray(rgb)
        depth = np.asarray(validdepth)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('HWC uint8 RGB required')
        if depth.shape != rgb.shape[:2]:
            raise ValueError('aligned measured depth/validity required')
        valid = depth.copy() if depth.dtype == bool else np.isfinite(depth) & (depth > 0)
        before = hashlib.sha256(rgb.tobytes()).hexdigest()
        started = time.perf_counter()
        result = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=self.confidence,
            iou=.7, imgsz=self.imgsz, device='cpu', half=False, retina_masks=True,
            verbose=False, save=False, stream=False, max_det=100)[0]
        elapsed = time.perf_counter() - started
        if tuple(result.orig_shape) != rgb.shape[:2]:
            raise RuntimeError('RGB/mask alignment mismatch')
        semantic = np.zeros(rgb.shape[:2], dtype=np.int16)
        instances = []
        if len(result.boxes):
            if result.masks is None:
                raise RuntimeError('segmentation model returned boxes without masks')
            masks = result.masks.data.detach().cpu().numpy() > .5
            if masks.shape[1:] != rgb.shape[:2]:
                raise RuntimeError('retina mask must have exact input dimensions')
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            # Global detector-confidence ownership avoids assigning overlapped
            # non-target objects a lower-confidence target label.
            owned = np.zeros(rgb.shape[:2], dtype=bool)
            for i in np.argsort(-scores, kind='stable'):
                name = self.prompts[labels[i]]
                measured = masks[i] & valid
                support = measured & ~owned
                owned |= measured
                cls = self.mapping.get(name)
                if cls is not None:
                    semantic[support] = cls
                instances.append(dict(instance_index=int(i), source_class_id=int(labels[i]),
                    source_label=name, class_key='yoloe_prompt:'+name,
                    detector_score=float(scores[i]), box_xyxy=boxes[i].tolist(),
                    v10_class_id=cls, mask=masks[i], valid_depth_mask=measured,
                    assigned_mask=support, mask_pixels=int(masks[i].sum()),
                    valid_depth_pixels=int(measured.sum()), assigned_pixels=int(support.sum()),
                    mask_sha256=hashlib.sha256(masks[i].tobytes()).hexdigest()))
        receipt = dict(schema='real_semantic_frontend/1', frame_id=frame_id,
            model_execution_performed=True, image_detection_alignment_verified=True,
            rgb_sha256=before, image_shape=list(rgb.shape), prompts=list(self.prompts),
            prior_profile=self.prior_profile, v10_mapping=dict(self.mapping), unknown_label=0,
            per_class_interpretation=dict(self.interpretations),
            semantic_prior_extended=self.prior_profile != 'strict_v10',
            core_formula_unchanged=True,
            class_semantics='uncalibrated prompt predictions; closed state and class accuracy not independently validated',
            box_is_segmentation_mask=False, mask_source='YOLOE retina instance segmentation',
            valid_depth_pixels=int(valid.sum()), semantic_pixels=int((semantic > 0).sum()),
            prediction_wall_seconds=elapsed, confidence_threshold=self.confidence,
            imgsz=self.imgsz, device='cpu', model_manifest=self.manifest,
            detections=[{k:v for k,v in item.items() if not isinstance(v,np.ndarray)} for item in instances],
            overlap_assignment='highest detector confidence owns each measured pixel including non-V10 classes',
            research_policy_changed=self.prior_profile != 'strict_v10')
        return dict(semantic=semantic, receipt=receipt, instances=instances)
