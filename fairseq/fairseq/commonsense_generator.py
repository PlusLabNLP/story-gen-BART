from fairseq.cometcommonsense import generate_conceptnet
from fairseq.data.encoders import gpt2_bpe_utils


def initialize():
	model , sampler , data_loader, text_encoder = generate_conceptnet.init()
	gpt_encoder = gpt2_bpe_utils.get_encoder("/nas/home/tuhinc/fairseq/encoder.json","/nas/home/tuhinc/fairseq/vocab.bpe")
	return model , sampler , data_loader, text_encoder,gpt_encoder


def convert_to_word(tgt_dict,s,gpt_encoder):
	converted = tgt_dict.string(s)
	tokens = [int(tok) if tok not in {'<unk>', '<mask>'} else tok for tok in converted.split()]
	converted = gpt_encoder.decode(tokens)
	print("converted is",converted)
	for sym in ['<A0>','<A1>','<A2>','<V>','</s>','<EOT>','<EOL>','#','ent']:
		converted = converted.replace(sym,'')
	for w in converted.split():
		if isdigit(w):
			converted = converted.replace(w,'')
	word = ""
	prev = None
	for c in converted:
		if c==' ' and c==prev:
			continue
		word = word+c
		prev = c
	converted = word
	if converted.startswith(' '):
		converted.lstrip()
	if converted.endswith(' '):
		converted.rstrip()
	return converted



def generate_next_word(tgt_dict, tokens, model, sampler,data_loader,text_encoder,gpt_encoder):
	
	y = tokens.tolist()
	count = 0
	for a in y:
		word = convert_to_word(tgt_dict,a,gpt_encoder)
		if word.count('#')==count+1:
			prefix = word.split('#')[-1]
			if prefix!=prev:
				next_word = generate_conceptnet.generate(prefix,model,sampler,data_loader,text_encoder)
				print("next_words are",next_word)
				prev = prefix
			count = count+1



