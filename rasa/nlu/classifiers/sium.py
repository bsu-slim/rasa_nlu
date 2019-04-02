from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# TODO: typing here and in other files
from typing import Any, Dict, List, Optional, Text

import pickle
from nltk import MaxentClassifier
from rasa.nlu.utils.sium_utils import SIUM
from rasa.nlu.components import Component
from rasa.nlu import utils
import os


from rasa.nlu.classifiers import INTENT_RANKING_LENGTH
from rasa.nlu.components import IncrementalComponent
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.tokenizers import Tokenizer, Token


class rasa_sium(IncrementalComponent):
    """A new component"""

    provides = ["intent", "intent_ranking", "entities", "tokens"]
    # TODO: require inc_iu_message
    requires = []
    defaults = {}

    # don't run unless these packages are installed on machine

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]]=None) -> None:

        super(rasa_sium, self).__init__(component_config)

        # Initialize sium instance
        self.sium = SIUM("rasa_sium")
        self.context = {}
        # Keep track of tokens, entities, and word_offset
        self.tokens = []
        self.extracted_entities = []
        self.word_offset = 0

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["nltk", "pickle"]

    def train(self,
              training_data: 'TrainingData',
              cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        for e in training_data.intent_examples:
            example = e.as_dict()
            intent = example["intent"]
            entities = example["entities"]
            for ent in entities:
                word = ent["value"].lower()
                prop = ent["entity"]
                for w in word.split():
                    self.sium.add_word_to_property(prop, {"word": w})
                if intent not in self.context:
                    self.context[intent] = {}
                self.context[intent][prop] = prop
        self.sium.train()

    # clears the internal state
    def new_utterance(self):
        self.sium.new_utt()
        self.extracted_entities = []
        self.tokens = []
        self.word_offset = 0

    # Here, we do the main processing in terms of evaluation
    # this will return intents, intent_rankings, entities, and
    # tokens in the message set. Tokenization is neccessary for
    # entity extraction in rasa, and we had to use our own in order
    # to keep our incremental framework possible. We use an extremely
    # simple whitespace tokenizer.
    def process(self, message: 'Message', **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""

        self.sium.set_context(self.context)

        # TODO: lowercase IU

        # The Latest IU is being appended to
        # "incr_edit_message" in the message,
        # so we grab last one out of that.
        inc_edit_message = message.get("incr_edit_message")
        new_iu = inc_edit_message[-1]
        # Extract into tuple of (word, type)
        # where type is either an "add" or "revoke".
        iu_word, iu_type = new_iu
        # If it's an add, we have to update our intents
        # and extract any entities if they meet our threshold.
        # We also have to keep track of our word offset for
        # the entities message.
        if iu_type is "add":
            self.tokens.append(Token(iu_word, self.word_offset))
            props, prop_dist = self.sium.add_word_increment({"word": iu_word})
            for p in props:
                # if we have a confidence of 0.5, then
                # add that entity
                if prop_dist.prob(p) > 0.5:
                    self.extracted_entities.append({
                        'start': self.word_offset,
                        'end': self.word_offset+len(iu_word)-1,
                        'value': iu_word, 'entity': p,
                        'confidence': prop_dist.prob(p),
                        'extractor': 'rasa_sium'
                    })
            self.word_offset += len(iu_word)
        elif iu_type is "revoke":
            # Need to undo everything above, remove tokens,
            # revoke word, remove extracted entities, subtract word_offset.
            self.word_offset -= len(iu_word)
            # Remove our latest token from our list.
            self.tokens.pop()
            # This is a bit more difficult, basically, if we have
            # our word show up in any extracted entities, then we
            # need to remove that entity from our list of entities.
            if self.extracted_entities:
                last_entity = self.extracted_entities[-1]
                if iu_word in last_entity.values():
                    self.extracted_entities.pop()
            self.sium.revoke()
        pred_intent, intent_ranks = self.__get_intents_and_ranks()
        message.set("intent", pred_intent, add_to_output=True)
        message.set("intent_ranking", intent_ranks)
        message.set("tokens", self.tokens)
        message.set("entities", self.extracted_entities, add_to_output=True)

    def __get_intents_and_ranks(self):
        # Get the current prediction state and the sum of all the
        # intent rankings this is needed to normalize the confidence
        #  values.
        intents_maxent_prob = self.sium.get_current_prediction_state()
        # Sum of all the intent ranking from maximum entropy
        # We need this to normalize confidences to sum to 1.
        int_rank_sum = sum(intents_maxent_prob.values())
        # Predict our intent, and calculate the normalized confidence
        # score for the intent.
        curr_pred_intent = self.sium.get_predicted_intent()[0]
        confidence = intents_maxent_prob[curr_pred_intent] / int_rank_sum
        pred_intent = {
            'name': curr_pred_intent,
            'confidence': confidence
        }
        # Rank and normalize the rest of the intents in terms of confidence.
        norm_rank = [{'name': intent,
                      'confidence': intents_maxent_prob[intent] / int_rank_sum}
                     for intent in intents_maxent_prob]
        return pred_intent, norm_rank

    # why might this be happening?
    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['rasa_sium'] = None,
             **kwargs: Any
             ) -> 'rasa_sium':
        file_name = meta.get("file")
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return utils.pycloud_unpickle(classifier_file)
        else:
            return cls(meta)

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        file_name = file_name + ".pkl"
        classifier_file = os.path.join(model_dir, file_name)
        utils.pycloud_pickle(classifier_file, self)
        return {"file": file_name}
