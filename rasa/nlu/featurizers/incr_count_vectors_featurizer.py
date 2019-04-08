import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.components import IncrementalComponent
from rasa.nlu.featurizers.count_vectors_featurizer import (
    CountVectorsFeaturizer
)


# Restart-Incremental wrapper for the CountVectorsFeaturizer
class IncrementalCVF(IncrementalComponent):
    name = "IncrementalCVF"

    """ Since this is a wrapper for the non-incremental
    CountVectorsFeaturizer to be used with our incremental
    EmbeddingIntentClassifier, we just need to take their
    provides, requires, and defaults.

    """
    provides = CountVectorsFeaturizer.provides
    requires = CountVectorsFeaturizer.requires
    defaults = CountVectorsFeaturizer.defaults

    @classmethod
    def required_packages(cls) -> List[Text]:
        return CountVectorsFeaturizer.required_packages()

    def __init__(self, component_config=None):
        super(IncrementalCVF, self).__init__(
            component_config)

        self.CVF = CountVectorsFeaturizer()
        self.utterance_so_far = ""
    
    def new_utterance(self) -> None:
        self.utterance_so_far = ""
    
    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig = None,
              **kwargs: Any) -> None:

        return self.CVF.train(training_data, cfg, **kwargs)

    # assuming not using spacy_doc or tokens, so just setting message.text
    def process(self, message: Message, **kwargs: Any) -> None:
        message.text = self.utterance_so_far
        print("!!!! ", message.text)
        return self.CVF.process(message, **kwargs)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:

        file_name = file_name + ".pkl"
        featurizer_file = os.path.join(model_dir, file_name)
        utils.pycloud_pickle(featurizer_file, self)
        return {"file": file_name}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional['IncrementalCVF'] = None,
             **kwargs: Any
             ) -> 'IncrementalCVF':

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            featurizer_file = os.path.join(model_dir, file_name)
            return utils.pycloud_unpickle(featurizer_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return IncrementalCVF(meta)
