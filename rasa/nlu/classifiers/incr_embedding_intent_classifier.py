import logging
import os
import io
import re
import pickle
from typing import Any, Dict, List, Optional, Text
import numpy as np

from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

from rasa.utils import train_utils

from rasa.nlu.components import IncrementalComponent
from rasa.nlu.classifiers.embedding_intent_classifier import (
    EmbeddingIntentClassifier
)

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
except ImportError:
    tf = None


# Restart-Incremental wrapper for the EmbeddingIntentClassifier
class IncrementalEIC(EmbeddingIntentClassifier,IncrementalComponent):
    name = "IncrementalEIC"

    """ Since this is a wrapper for the non-incremental
    EmbeddingIntentClassifier to be used with our incremental
    EmbeddingIntentClassifier, we just need to take its
    provides, requires, and defaults.

    """

    def new_utterance(self) -> None:
        self.prev_intent_and_rank = []

   
    def process(self, message: Message, **kwargs: Any) -> None:
        iu_list = message.get('iu_list')
        last_iu = iu_list[-1]
        iu_word, iu_type = last_iu
        if iu_type == "add":
            if not hasattr(self, 'prev_intent_and_rank'): self.new_utterance() # hack
            self.prev_intent_and_rank.append({"intent": message.get("intent"),
                                              "intent_ranking": message.get("intent_ranking")})
            return super(IncrementalEIC,self).process(message, **kwargs)
        elif iu_type == "revoke":
            return self._revoke(message)
        else:
            logger.error("incompatible iu type, expected 'add' or 'revoke',"
                         " got '" + iu_type + "'")

    def _revoke(self, message):
        # revoke on empty should do nothing
        if not hasattr(self, 'prev_intent_and_rank'):
            return
        else:
            if len(self.prev_intent_and_rank) > 0:
                prev_state = self.prev_intent_and_rank.pop()
                message.set("intent", prev_state["intent"])
                message.set("intent_ranking", prev_state["intent_ranking"])

    