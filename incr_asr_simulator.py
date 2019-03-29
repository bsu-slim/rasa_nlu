from rasa_nlu.model import IncrementalInterpreter
import json
import random

# This is a very rudementary incremental ASR simulator.
# give an entire utterance, and it splits it into
# separate iu components, with a 20% chance of giving
# a random add and revoke from the list below. It prints
# the result of the parse at each step.

interpreter = IncrementalInterpreter.load("./models/current/sium")
random_revokes = ["restaurant", "italian", "playlist", "thai"
                  "store", "four", "movie", "showtime"]

while 1:
    interpreter.new_utterance()
    message = input("Type full utterance:\n")
    message_sep = message.split()
    message_to_pass = []
    for word in message_sep:
        message_to_pass.append((word, "add"))
        # 20 percent incorrect rate, add
        # a word then revoke it
        if random.random() < 0.2:
            random_word = random.choice(random_revokes)
            message_to_pass.append((random_word, "add"))
            message_to_pass.append((random_word, "revoke"))

    for iu in message_to_pass:
        print("IU: ", iu)
        result = interpreter.parse_incremental(iu)
        print("After parse: ", json.dumps(result, indent=2))
