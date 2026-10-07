"""Fill missing processor configs before starting the offline model process."""
import hashlib
import json
import shutil
from pathlib import Path

NAMES=('processor_config.json','preprocessor_config.json','video_preprocessor_config.json')
BASE='google/gemma-4-12b-it'

def ensure_processors(model_dir):
    """Offline-only check. The serving process never touches the network (evaluators run it
    with networking disabled). Missing optional processor configs are reported, not fetched;
    prepare_model.sh is the only step that downloads anything. The Gemma 4 INT4 checkpoint
    serves text decisions without preprocessor/video configs (verified on vLLM 0.31)."""
    p=Path(model_dir)
    missing=[n for n in NAMES if not (p/n).is_file()]
    if missing:
        print('[model_setup] optional processor configs absent (not needed for text): '+', '.join(missing),flush=True)
    return missing
