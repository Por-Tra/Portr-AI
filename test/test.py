from tokenizer.tokenizer import Tokenizer


tok = Tokenizer.load("tokenizer/token/bpe_model.json")
tokens = tok.tokenize_text("votre texte à tokeniser ici")