from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import datetime
import logging
import os

# for revoke testing
# todo: remove this as cleanup after asr inputs
import random

from builtins import object
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

import rasa_nlu
from rasa_nlu import components, utils, config
from rasa_nlu.components import Component, ComponentBuilder
from rasa_nlu.config import RasaNLUModelConfig, override_defaults
from rasa_nlu.persistor import Persistor
from rasa_nlu.training_data import TrainingData, Message
from rasa_nlu.utils import create_dir, write_json_to_file

logger = logging.getLogger(__name__)

MINIMUM_COMPATIBLE_VERSION = "0.13.0a2"


class InvalidProjectError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class UnsupportedModelError(Exception):
    """Raised when a model is to old to be loaded.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class Metadata(object):
    """Captures all information about a model to load and prepare it."""

    @staticmethod
    def load(model_dir):
        # type: (Text) -> 'Metadata'
        """Loads the metadata from a models directory.

        Args:
            model_dir (str): the directory where the model is saved.
        Returns:
            Metadata: A metadata object describing the model
        """
        try:
            metadata_file = os.path.join(model_dir, 'metadata.json')
            data = utils.read_json_file(metadata_file)
            return Metadata(data, model_dir)
        except Exception as e:
            abspath = os.path.abspath(os.path.join(model_dir, 'metadata.json'))
            raise InvalidProjectError("Failed to load model metadata "
                                      "from '{}'. {}".format(abspath, e))

    def __init__(self, metadata, model_dir):
        # type: (Dict[Text, Any], Optional[Text]) -> None

        self.metadata = metadata
        self.model_dir = model_dir

    def get(self, property_name, default=None):
        return self.metadata.get(property_name, default)

    @property
    def component_classes(self):
        if self.get('pipeline'):
            return [c.get("class") for c in self.get('pipeline', [])]
        else:
            return []

    def for_component(self, name, defaults=None):
        return config.component_config_from_pipeline(name,
                                                     self.get('pipeline', []),
                                                     defaults)

    @property
    def language(self):
        # type: () -> Optional[Text]
        """Language of the underlying model"""

        return self.get('language')

    def persist(self, model_dir):
        # type: (Text) -> None
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update({
            "trained_at": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "rasa_nlu_version": rasa_nlu.__version__,
        })

        filename = os.path.join(model_dir, 'metadata.json')
        write_json_to_file(filename, metadata, indent=4)


class Trainer(object):
    """Trainer will load the data and train all components.

    Requires a pipeline specification and configuration to use for
    the training."""

    # Officially supported languages (others might be used, but might fail)
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(self,
                 cfg,  # type: RasaNLUModelConfig
                 component_builder=None,  # type: Optional[ComponentBuilder]
                 skip_validation=False  # type: bool
                 ):
        # type: (...) -> None

        self.config = cfg
        self.skip_validation = skip_validation
        self.training_data = None  # type: Optional[TrainingData]

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in
            # a new builder. hence, no components are reused.
            component_builder = components.ComponentBuilder()

        # Before instantiating the component classes, lets check if all
        # required packages are available
        if not self.skip_validation:
            components.validate_requirements(cfg.component_names)

        # build pipeline
        self.pipeline = self._build_pipeline(cfg, component_builder)

    @staticmethod
    def _build_pipeline(cfg, component_builder):
        # type: (RasaNLUModelConfig, ComponentBuilder) -> List
        """Transform the passed names of the pipeline components into classes"""
        pipeline = []

        # Transform the passed names of the pipeline components into classes
        for component_name in cfg.component_names:
            component = component_builder.create_component(
                    component_name, cfg)
            pipeline.append(component)

        return pipeline

    def train(self, data, **kwargs):
        # type: (TrainingData) -> Interpreter
        """Trains the underlying pipeline using the provided training data."""

        self.training_data = data

        context = kwargs  # type: Dict[Text, Any]

        for component in self.pipeline:
            updates = component.provide_context()
            if updates:
                context.update(updates)

        # Before the training starts: check that all arguments are provided
        if not self.skip_validation:
            components.validate_arguments(self.pipeline, context)

        # data gets modified internally during the training - hence the copy
        working_data = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info("Starting to train component {}"
                        "".format(component.name))
            component.prepare_partial_processing(self.pipeline[:i], context)
            updates = component.train(working_data, self.config,
                                      **context)
            logger.info("Finished training component.")
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context)

    def persist(self, path, persistor=None, project_name=None,
                fixed_model_name=None):
        # type: (Text, Optional[Persistor], Text) -> Text
        """Persist all components of the pipeline to the passed path.

        Returns the directory of the persisted model."""

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {
            "language": self.config["language"],
            "pipeline": [],
        }

        if project_name is None:
            project_name = "default"

        if fixed_model_name:
            model_name = fixed_model_name
        else:
            model_name = "model_" + timestamp

        path = config.make_path_absolute(path)
        dir_name = os.path.join(path, project_name, model_name)

        create_dir(dir_name)

        if self.training_data:
            metadata.update(self.training_data.persist(dir_name))

        for component in self.pipeline:
            update = component.persist(dir_name)
            component_meta = component.component_config
            if update:
                component_meta.update(update)
            component_meta["class"] = utils.module_path_from_object(component)
            metadata["pipeline"].append(component_meta)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.persist(dir_name, model_name, project_name)
        logger.info("Successfully saved model into "
                    "'{}'".format(os.path.abspath(dir_name)))
        return dir_name


class Interpreter(object):
    """Use a trained pipeline of components to parse text messages."""

    # Defines all attributes (& default values)
    # that will be returned by `parse`
    @staticmethod
    def default_output_attributes():
        return {"intent": {"name": None, "confidence": 0.0}, "entities": []}

    @staticmethod
    def ensure_model_compatibility(metadata, version_to_check=None):
        from packaging import version

        if version_to_check is None:
            version_to_check = MINIMUM_COMPATIBLE_VERSION

        model_version = metadata.get("rasa_nlu_version", "0.0.0")
        if version.parse(model_version) < version.parse(version_to_check):
            raise UnsupportedModelError(
                "The model version is to old to be "
                "loaded by this Rasa NLU instance. "
                "Either retrain the model, or run with"
                "an older version. "
                "Model version: {} Instance version: {}"
                "".format(model_version, rasa_nlu.__version__))

    @staticmethod
    def load(model_dir, component_builder=None, skip_validation=False):
        """Create an interpreter based on a persisted model.

        Args:
            model_dir (str): The path of the model to load
            component_builder (ComponentBuilder): The
                :class:`ComponentBuilder` to use.

        Returns:
            Interpreter: An interpreter that uses the loaded model.
        """

        model_metadata = Metadata.load(model_dir)

        Interpreter.ensure_model_compatibility(model_metadata)
        return Interpreter.create(model_metadata,
                                  component_builder,
                                  skip_validation)

    @staticmethod
    def create(model_metadata,  # type: Metadata
               component_builder=None,  # type: Optional[ComponentBuilder]
               skip_validation=False  # type: bool
               ):
        # type: (...) -> Interpreter
        """Load stored model and components defined by the provided metadata."""

        context = {}

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result
            # in a new builder. hence, no components are reused.
            component_builder = components.ComponentBuilder()

        pipeline = []

        # Before instantiating the component classes,
        # lets check if all required packages are available
        if not skip_validation:
            components.validate_requirements(model_metadata.component_classes)

        for component_name in model_metadata.component_classes:
            component = component_builder.load_component(
                    component_name, model_metadata.model_dir,
                    model_metadata, **context)
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. "
                                "{}".format(component.name, e))

        return Interpreter(pipeline, context, model_metadata)

    def __init__(self, pipeline, context, model_metadata=None):
        # type: (List[Component], Dict[Text, Any], Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.model_metadata = model_metadata

    def parse(self, text, time=None, only_output_properties=True):
        # type: (Text) -> Dict[Text, Any]
        """Parse the input text, classify it and return pipeline result.

        The pipeline result usually contains intent and entities."""

        if not text:
            # Not all components are able to handle empty strings. So we need
            # to prevent that... This default return will not contain all
            # output attributes of all components, but in the end, no one
            # should pass an empty string in the first place.
            output = self.default_output_attributes()
            output["text"] = ""
            return output

        message = Message(text, self.default_output_attributes(), time=time)

        for component in self.pipeline:
            component.process(message, **self.context)

        output = self.default_output_attributes()
        output.update(message.as_dict(
                only_output_properties=only_output_properties))
        return output

# Programmatically with an explicitly listed incremental component.
# Further documentation needed (TODO)


class Incremental_Interpreter(Interpreter):

    # must override load and create to return Incremental_Interpreter
    # rather than Interpreter
    @staticmethod
    def load(model_dir, component_builder=None, skip_validation=False):
        """Create an interpreter based on a persisted model.

        Args:
            model_dir (str): The path of the model to load
            component_builder (ComponentBuilder): The
                :class:`ComponentBuilder` to use.

        Returns:
            Interpreter: An interpreter that uses the loaded model.
        """

        model_metadata = Metadata.load(model_dir)

        Incremental_Interpreter.ensure_model_compatibility(model_metadata)
        return Incremental_Interpreter.create(model_metadata,
                                              component_builder,
                                              skip_validation)

    @staticmethod
    def create(model_metadata,  # type: Metadata
               component_builder=None,  # type: Optional[ComponentBuilder]
               skip_validation=False  # type: bool
               ):
        # type: (...) -> Interpreter
        """Load stored model and components defined by the provided metadata."""

        context = {}

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result
            # in a new builder. hence, no components are reused.
            component_builder = components.ComponentBuilder()

        pipeline = []

        # Before instantiating the component classes,
        # lets check if all required packages are available
        if not skip_validation:
            components.validate_requirements(model_metadata.component_classes)

        for component_name in model_metadata.component_classes:
            component = component_builder.load_component(
                    component_name, model_metadata.model_dir,
                    model_metadata, **context)
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. "
                                "{}".format(component.name, e))

        return Incremental_Interpreter(pipeline, context, model_metadata)

    # Overriding the init function to make message a variable contained in self
    # so that it can be preserved across multiple incremental_parse calls, 
    # self.message is initialized in init, and get modified with each parse
    # call until it is eventually cleared when new_utterance is called
    def __init__(self, pipeline, context, model_metadata=None):
        super().__init__(pipeline, context, model_metadata)
        # TODO: assert that every compoenent in the pipeline is incremental
        self.message = Message(text="")

    # Call this function when creating up a new utterance
    # this will tell the incremental components to clear their
    # internal states and start clean.
    def new_utterance(self):
        for component in self.pipeline:
            component.new_utterance()
        self.message = Message(text="")

    # def revoke_word(self):
    #     prev_iu = self.message.get('incr_edit_message')
    #     if prev_iu:
    #         prev_iu = prev_iu[-1]
    #         revoke_iu = (prev_iu[0], "revoke")
    #         self.message.get('incr_edit_message').append(revoke_iu)
    #         for component in self.pipeline:
    #             component.process(self.message, **self.context)
    #     return prev_iu

    # here, parse will be preserved but be breaking up the text into individual
    # words, then fed into the incremental component. This way, cmd-line evaluation 
    # can be preserved without many changes, and we can still evaluate an incremental
    # component from cmd-line.
    def parse(self, text, time=None, only_output_properties=True):
        self.new_utterance()
        for word in text.split():
            iu = (word, "add")
            self.parse_incremental(iu, time)
        output = self.default_output_attributes()
        output.update(self.message.as_dict(
                      only_output_properties=only_output_properties))
        return output

    # new_utterance should be called before each new utterance, and that component
    # will be responsible for clearing its internal state. However, the message
    # is responsible for clearing/restarting the message object
    # iu parameter is a tuple in the form of (word, type), where word is the 
    # word being passed by the asr, and type is of either "add" or 
    # "revoke". This just puts the iu on the message bus, and the 
    # component has the responsibility of handling adds or revokes.
    def parse_incremental(self, iu, time=None, only_output_properties=True):
        if not iu:
            # Not all components are able to handle empty strings. So we need
            # to prevent that... This default return will not contain all
            # output attributes of all components, but in the end, no one
            # should pass an empty string in the first place.
            output = self.default_output_attributes()
            output["text"] = ""
            return output

        # Initialize our incr_edit_message if this is the first
        # call to it.
        if(self.message.get('incr_edit_message') is None):
            self.message.set('incr_edit_message', list())

        self.message.get('incr_edit_message').append(iu)

        for component in self.pipeline:
            component.process(self.message, **self.context)
        output = self.default_output_attributes()
        output.update(self.message.as_dict(
                only_output_properties=only_output_properties))
        return output

