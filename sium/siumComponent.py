from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
from nltk import MaxentClassifier
from rasa_nlu.sium import SIUM
from rasa_nlu.components import Component
from rasa_nlu import utils
import os

from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Incremental_Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.tokenizers import Tokenizer, Token


class RASA_SIUM(Incremental_Component):
    """A new component"""

    # Name of the component to be used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    name = "sium"

    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = ["intent", "intent_ranking", "entities", "tokens"]

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.

    # need to talk to Dr.k about what it requires
    requires = []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = None

    # don't run unless these packages are installed on machine
    @classmethod
    def required_packages(cls):
        return ["nltk", "pickle"]

    def __init__(self, component_config=None):
        self.sium = SIUM("rasa-sium")
        self.context = {}
        self.tokens = []
        self.extracted_entities = []
        self.word_offset = 0
        self.component_config = component_config

    def train(self, training_data, cfg, **kwargs):
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

    # here, we do the main processing in terms of evaluation
    # this will return intents, intent_rankings, entities, and
    # tokens in the message set. Tokenization is neccessary for 
    # entity extraction in rasa, and we had to use our own in order
    # to keep our incremental framework possible. We use an extremely
    # simple whitespace tokenizer 
    def process(self, message, **kwargs):
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        
        # does this need to be here every call?
        self.sium.set_context(self.context)
        
        # latest IU is being appended to "incr_edit_message", so grab last one out of that
        inc_edit_message = message.get("incr_edit_message")
        # latest IU being added, for now, assume all add
        new_iu = inc_edit_message[-1]

        iu_word, iu_type = new_iu
        if iu_type is "add":
            self.tokens.append(Token(iu_word, self.word_offset))
            props, prop_dist = self.sium.add_word_increment({"word": iu_word})
            for p in props:
                # todo: multi-word entities, threshold adjustments
                if prop_dist.prob(p) > 0.5:
                    self.extracted_entities.append({'start': self.word_offset, 
                        'end': self.word_offset+len(iu_word)-1, 'value': 
                        iu_word, 'entity': p, 'confidence': prop_dist.prob(p), 
                        'extractor': 'sium'})
            self.word_offset += len(iu_word)
        elif iu_type is "revoke":
            # need to undo everythin above, remove tokens, revoke word, remove extracted entities, subtract word_offset
            # correct word_offset since we are removing this word
            self.word_offset -= len(iu_word)
            # remove our token with that word from our list of tokens
            self.tokens.pop()
            # this is a bit more difficult, basically, if we have 
            # our word show up in any extracted entities, then we
            # need to remove that entity from our list of entities
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
        # get the current prediction state and the sum of all the intent rankings
        # this is needed to normalize the confidence values.
        intents_maxent_prob = self.sium.get_current_prediction_state()
        intent_ranking_sum = sum(intents_maxent_prob.values())
        # predict our intent, and calculate the normalized confidence score for it
        pred_intent = {'name': self.sium.get_predicted_intent()[0], 
            'confidence': intents_maxent_prob[self.sium.get_predicted_intent()[0]]/intent_ranking_sum}
        # rank the rest of the intents in terms of confidence.
        norm_intent_ranking = [{'name': intent,
                                   'confidence': intents_maxent_prob[intent]/intent_ranking_sum}
                                  for intent in intents_maxent_prob]  
        return pred_intent, norm_intent_ranking      

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> SklearnIntentClassifier

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", 'sium')
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return utils.pycloud_unpickle(classifier_file)
        else:
            return cls(meta)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory."""

        classifier_file = os.path.join(model_dir, 'sium')
        utils.pycloud_pickle(classifier_file, self)
        return {'classifier_file': 'sium'}
