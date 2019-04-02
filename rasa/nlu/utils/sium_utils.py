'''
Created on Aug 21, 2018

@author: ckennington
'''
from nltk import MaxentClassifier
import pickle
from operator import itemgetter


class SIUM:
    '''
        name: name of model for persistance and loading

        to train:
        add observations, then call train() then persist()

        to evaluate:
        load() model, then call add_increment for each word in an utterance.
        Call new_utt() to start a new utterance
    '''
    def __init__(self, model_name):
        self.prop_intent = {}
        self.word_prop = {}
        self.model = {}
        self.model_name = model_name
        self.context = {}
        self.current_utt = {}
        self.previous_states = []

    def add_word_to_property(self, prop, feats):
        '''
        feats is a dict where the keys are prop types, and values are
        the props themselves
        '''
        if prop not in self.word_prop:
            self.word_prop[prop] = list()
        self.word_prop[prop].append(feats)

    def load_model(self):
        with open('{}.pickle'.format(self.model_name), 'rb') as handle:
            self.model = pickle.load(handle)

    def persist_model(self):
        with open('{}.pickle'.format(self.model_name), 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_word_increment(self, feats):
        # SIUM: sum_{r /in R} P(R=r|U) P(R=r|I)
        self.previous_states.append({
            "current_utt": self.current_utt.copy(),
            "context": self.context.copy()
        })

        # get P(R|U)
        dist = self.model.prob_classify(feats)
        props = dist.samples()
        # do the summation
        p_u = {i: 0 for i in self.context}
        for r in props:
            p = dist.prob(r)
            for i in self.context:
                # this check is essentially P(R|I)
                if r in self.context[i].values():
                    # normalize for number of props, otherwise objects
                    # with more properties get higher overall probs
                    p_u[i] = p_u[i] + p / len(self.context[i])
        # first time, there is no prior
        if self.current_utt == {}:
            self.current_utt = p_u
            return props, dist
            # return self.current_utt

        for i in self.current_utt:
            # this combines P(U) with the rest of the stuff
            self.current_utt[i] = self.current_utt[i] * p_u[i]

        return props, dist

    # in case of revert, remove latest word increment and restore
    # revokes the previous work and restores SIUM to the state it was
    # in before
    def revoke(self):
        # make sure we are not revoking on empty
        if not self.previous_states:
            pass
        else:
            prev_state = self.previous_states.pop()
            self.current_utt = prev_state['current_utt']
            self.context = prev_state['context']

    def get_current_prediction_state(self):
        return self.current_utt

    def get_predicted_intent(self):
        return max(self.get_current_prediction_state().items(),
                   key=itemgetter(1))

    def set_context(self, context):
        '''
        context should have the following structure (i.e., a dict of dicts):

        {id1:{prop1type:prop1, prop2type:prop2, ... propNtype:propN)},
         ...
         idM:{prop1type:prop1, prop2type:prop2, ... propNtype:propN)}}

        '''
        self.context = context

    def train(self):
        feature_set = list()
        for prop in self.word_prop:
            for feats in self.word_prop[prop]:
                feature_set.append((feats, prop))
        self.model = MaxentClassifier.train(feature_set, "gis", max_iter=10)

    def new_utt(self):
        self.context = {}
        self.current_utt = {}
        self.previous_states = []
