import re
from typing import Any, List, Text, Optional, Dict

from rasa.nlu.components import IncrementalComponent
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import Message, TrainingData


class IncrementalWhitespaceTokenizer(Tokenizer, IncrementalComponent):

    provides = ["tokens"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]]=None):

        super(IncrementalWhitespaceTokenizer, self).__init__(component_config)
        self.offset = 0
        self.tokens = []
        self.WST = WhitespaceTokenizer()

    def new_utterance(self):
        self.offset = 0
        self.tokens = []

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.WST.tokenize(text=example.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        iu_list = message.get('iu_list')
        last_iu = iu_list[-1]
        iu_word, iu_type = last_iu
        if iu_type == "add":
            token = self.WST.tokenize(iu_word)
            if token:
                token = token[0]
                token.offset = self.offset
                token.end = token.offset + token.end
                self.offset += (token.end - token.offset + 1)
                self.tokens.append(token)
        elif iu_type == "revoke":
            removed = self.tokens.pop()
            self.offset = removed.offset
        else:
            logger.error("incompatible iu type, expected 'add' or 'revoke',"
                         " got '" + iu_type + "'")
        message.set("tokens", self.tokens)
