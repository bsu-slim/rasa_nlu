from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.components import IncrementalComponent

class IncrementalMitieEntityExtractor(MitieEntityExtractor, IncrementalComponent):
    pass