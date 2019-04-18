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

from rasa.nlu.components import IncrementalComponent
from rasa.nlu.classifiers.embedding_intent_classifier import (
    EmbeddingIntentClassifier
)

try:
    import tensorflow as tf
except ImportError:
    tf = None


# Restart-Incremental wrapper for the EmbeddingIntentClassifier
class IncrementalEIC(IncrementalComponent):
    name = "IncrementalEIC"

    """ Since this is a wrapper for the non-incremental
    EmbeddingIntentClassifier to be used with our incremental
    EmbeddingIntentClassifier, we just need to take its
    provides, requires, and defaults.

    """
    provides = EmbeddingIntentClassifier.provides
    requires = EmbeddingIntentClassifier.requires
    defaults = EmbeddingIntentClassifier.defaults

    @classmethod
    def required_packages(cls) -> List[Text]:
        reqs = EmbeddingIntentClassifier.required_packages()
        reqs.append("numpy")
        return reqs

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 inv_intent_dict: Optional[Dict[int, Text]] = None,
                 encoded_all_intents: Optional[np.ndarray] = None,
                 session: Optional['tf.Session'] = None,
                 graph: Optional['tf.Graph'] = None,
                 message_placeholder: Optional['tf.Tensor'] = None,
                 intent_placeholder: Optional['tf.Tensor'] = None,
                 similarity_op: Optional['tf.Tensor'] = None,
                 word_embed: Optional['tf.Tensor'] = None,
                 intent_embed: Optional['tf.Tensor'] = None
                 ) -> None:
        super(IncrementalEIC, self).__init__(
            component_config)

        self.EIC = EmbeddingIntentClassifier(component_config,
                                             inv_intent_dict,
                                             encoded_all_intents,
                                             session,
                                             graph,
                                             message_placeholder,
                                             intent_placeholder,
                                             similarity_op,
                                             word_embed,
                                             intent_embed)
        # storing a list of dicts, containing the intents
        # and rankings from the Message object to revert 
        # to on a revoke.
        self.prev_intent_and_rank = []

    def new_utterance(self) -> None:
        self.prev_intent_and_rank = []

    def train(self,
              training_data: TrainingData,
              cfg: RasaNLUModelConfig = None,
              **kwargs: Any) -> None:
        return self.EIC.train(training_data, cfg, **kwargs)

    def process(self, message: Message, **kwargs: Any) -> None:
        iu_list = message.get('iu_list')
        last_iu = iu_list[-1]
        iu_word, iu_type = last_iu
        if iu_type == "add":
            self.prev_intent_and_rank.append({"intent": message.get("intent"),
                                              "intent_ranking": message.get("intent_ranking")})
            return self.EIC.process(message, **kwargs)
        elif iu_type == "revoke":
            return self._revoke(message)
        else:
            logger.error("incompatible iu type, expected 'add' or 'revoke',"
                         " got '" + iu_type + "'")

    def _revoke(self, message):
        # revoke on empty should do nothing
        if not self.prev_intent_and_rank:
            return
        else:
            prev_state = self.prev_intent_and_rank.pop()
            message.set("intent", prev_state["intent"])
            message.set("intent_ranking", prev_state["intent_ranking"])

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        return self.EIC.persist((file_name) + "_incr", model_dir)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional['IncrementalEIC'] = None,
             **kwargs: Any
             ) -> 'IncrementalEIC':
        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + '.meta')

                saver.restore(sess, checkpoint)

                a_in = tf.get_collection('message_placeholder')[0]
                b_in = tf.get_collection('intent_placeholder')[0]

                sim_op = tf.get_collection('similarity_op')[0]

                word_embed = tf.get_collection('word_embed')[0]
                intent_embed = tf.get_collection('intent_embed')[0]

            with io.open(os.path.join(
                    model_dir,
                    file_name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    file_name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            return cls(
                component_config=meta,
                inv_intent_dict=inv_intent_dict,
                encoded_all_intents=encoded_all_intents,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                intent_placeholder=b_in,
                similarity_op=sim_op,
                word_embed=word_embed,
                intent_embed=intent_embed
            )
        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
