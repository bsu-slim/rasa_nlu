import logging
import os
import re
from typing import Any, Dict, List, Optional, Text
import numpy as np

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
    EmbeddingIntentClassifier, we just need to take its
    provides, requires, and defaults.

    """
    provides = CountVectorsFeaturizer.provides
    requires = CountVectorsFeaturizer.requires
    defaults = CountVectorsFeaturizer.defaults

    @classmethod
    def required_packages(cls) -> List[Text]:
        reqs = CountVectorsFeaturizer.required_packages()
        reqs.append("numpy")
        return reqs

    def __init__(self, component_config=None):
        super(IncrementalCVF, self).__init__(
            component_config)

        self.CVF = CountVectorsFeaturizer()

    # we don't have anything to clear since our featuers are storeed
    # in the Message, which the IncrementalInterpreter clears.
    def new_utterance(self) -> None:
        return

    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig = None,
              **kwargs: Any) -> None:

        return self.CVF.train(training_data, cfg, **kwargs)

    # Similar to Featurizer's _combine_with_existing_text_features
    # Except we are doing a vector sum instead of array stack. This
    # is because we're adding the new features of that word in particular
    # rather than entire utterances side by side.
    def _add_text_features(self, message, additional_features):
        if message.get("text_features") is not None:
            return np.add(message.get("text_features"), additional_features)
        else:
            return additional_features

    # On revoke, remove the word's features from the vector
    def _sub_text_features(self, message, to_sub):
        if message.get("text_features") is not None:
            return np.subtract(message.get("text_features"), to_sub)
        else:
            logger.error("Nothing in text features, cannot subtract")

    # assuming not using spacy_doc or tokens, so just setting message.text
    def process(self, message: Message, **kwargs: Any) -> None:
        iu_list = message.get('iu_list')
        last_iu = iu_list[-1]
        iu_word, iu_type = last_iu
        if iu_type == "add":
            bag = self.CVF.vect.transform([iu_word]).toarray().squeeze()
            return message.set("text_features",
                               self._add_text_features(message, bag))
        elif iu_type == "revoke":
            return self._revoke(message, iu_word)
        else:
            logger.error("incompatible iu type, expected 'add' or 'revoke',"
                         " got '" + iu_type + "'")

    # TODO: can we just subtract the vector instead of
    # storing previous features?
    def _revoke(self, message, word):
        # revoke on empty should do nothing
        if message.get("text_features") is not None:
            return
        else:
            bag = self.CVF.vect.transform([word]).toarray().squeeze()
            return message.set("text_features",
                               self._sub_text_features(message, bag))

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
