UNK_WORD = '[UNK]'
SEP_WORD = '</s>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_WORD = '[PAD]'

# Globals for import elsewhere
DELIMITERS = {"<EOL>", "#", "<EOT>", "<P>", SEP_WORD, BOS_WORD, EOS_WORD, PAD_WORD}
SRL_TOKS = {"<A0>", "<A1>", "<A2>", "<V>"}
ENTITY_TOKS = {"</ent>", "<ent>"}
SPECIAL_CHARACTERS = DELIMITERS | SRL_TOKS | ENTITY_TOKS
BERT_CHARS = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"}
GPT2_CHARS = {"<|endoftext|>"}
