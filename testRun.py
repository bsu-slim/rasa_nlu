from rasa_nlu.model import Interpreter
import json

interpreter = Interpreter.load("./models/current/sium")
while 1:
    message = input("Type Query:\n")
    result = interpreter.parse(message)
    print(json.dumps(result, indent=2))
