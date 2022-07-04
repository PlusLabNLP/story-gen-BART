import spacy

def spacy_word_token(text):
    special_chars = {'<EOL>', '<EOT>', '<eos>', '</s>', '#', '<P>', '``', '\'\''} # harcoded list of special characters not to touch
    nlp = None  # this is for spacy use but it is slow so we don't load it if not necessary
    spacy_model = 'en_core_web_lg'
    nlp = spacy.load(spacy_model)
    # Need to special case all special chars for tokenization
    for key in special_chars:
        nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])
    doc = nlp(text)
    token_list = [t.text for t in doc]
    print(token_list)
    return token_list

if __name__ == '__main__':
    spacy_word_token("</s> <P> <P> `` Rob ? '' </s> <P> <P> Rob Hall , 97 , Corporal in the 50-100 army grins , as the situation turns from life or death struggle , to a meeting of two college friends . </s> He lets go of Marguerian 's collar . </s> <P> <P> `` Holy shit Clancy , you 're the last person I expected to see here .'' </s> <P> <P> `` Yeah '' </s> <P> <P>")