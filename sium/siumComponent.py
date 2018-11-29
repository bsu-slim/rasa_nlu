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


class RASA_SIUM(Component):
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
    provides = ["intents", "entities"]

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.

    #need to talk to Dr.k about what it requires
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
    #TODO: is SIUM language specific?
    language_list = None

    
    #don't run unless these packages are installed on machine
    @classmethod
    def required_packages(cls):
        return ["nltk", "pickle"]

    def __init__(self, component_config=None):
        self.sium = SIUM('rasa-sium')
        self.context = {}
        self.component_config=component_config


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
        print("!!!!!!!!!!!")
        for e in training_data.intent_examples:
            example = e.as_dict()
            intent = example['intent']
            entities = example['entities']
            for ent in entities:
                word = ent['value'].lower()
                prop = ent['entity']
                for w in word.split():
                    self.sium.add_word_to_property(prop, {'word':w})
                if intent not in self.context:
                    self.context[intent] = {}
                self.context[intent][prop] = prop
        print("!!!!!!!!!!!!!!")
        self.sium.train()

    
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
        self.sium.new_utt()
        self.sium.set_context(self.context)
        print(message.text)
        for word in message.text.lower().split():
            props,prop_dist = self.sium.add_word_increment({'word':word})
        print(self.sium.get_current_prediction_state())
        for p in props:
            print(p, prop_dist.prob(p))
        

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
        return {"classifier_file": 'sium'}
