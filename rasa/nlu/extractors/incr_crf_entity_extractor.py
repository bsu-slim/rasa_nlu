import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.components import IncrementalComponent
from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn_crfsuite


class IncrementalCRFEntityExtractor(EntityExtractor, IncrementalComponent):

    provides = ["entities"]

    requires = ["tokens"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]]=None,
                 ent_tagger: Optional[Dict[Text, Any]]=None) -> None:

        super(IncrementalCRFEntityExtractor, self).__init__(component_config)

        self.CRFEE = CRFEntityExtractor(component_config, ent_tagger)
        self.prev_ents = []

    def new_utterance(self):
        self.prev_ents = []

    @classmethod
    def required_packages(cls):
        return ["sklearn_crfsuite", "sklearn"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        self.CRFEE.train(training_data, config, **kwargs)

    def process(self, message: Message, **kwargs: Any) -> None:
        iu_list = message.get('iu_list')
        last_iu = iu_list[-1]
        iu_word, iu_type = last_iu
        # TODO: inefficient right now, we are always storing
        # previous state, even if a new entity hasn't been
        # added

        # This will not work with multiple extractors
        if iu_type == "add":
            extracted = self.add_extractor_name(
                self.CRFEE.extract_entities(message))
            message.set("entities", extracted, add_to_output=True)
            self.prev_ents.append(message.get("entities"))
        elif iu_type == "revoke":
            if len(self.prev_ents) > 0:
                prev_ent = self.prev_ents.pop()
                message.set("entities", prev_ent,
                            add_to_output=True)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional['IncrementalCRFEntityExtractor'] = None,
             **kwargs: Any
             ) -> 'IncrementalCRFEntityExtractor':
        from sklearn.externals import joblib

        file_name = meta.get("file")
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            ent_tagger = joblib.load(model_file)
            return cls(meta, ent_tagger)
        else:
            return cls(meta)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        return self.CRFEE.persist((file_name) + "_incr", model_dir)
