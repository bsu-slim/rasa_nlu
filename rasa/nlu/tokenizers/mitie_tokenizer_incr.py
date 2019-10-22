from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.components import IncrementalComponent

class IncrementalMitieTokenizer(MitieTokenizer, IncrementalComponent):

    pass


    #todo: implement new_utterance()