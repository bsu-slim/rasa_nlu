from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.components import IncrementalComponent

class IncrementalEntitySynonymMapper(EntitySynonymMapper, IncrementalComponent):

    pass