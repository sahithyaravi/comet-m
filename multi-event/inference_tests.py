import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import sys
from collections import defaultdict
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from comet_atomic2020_bart.utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
import random
import pandas as pd
import re
import numpy as np


BATCH_SIZE = 1


nlp = spacy.load('en_core_web_md')
ignore_verbs = ["are", "is", "were", "was"]

def replace(p, subst, test_str):
    result = re.sub(p, subst, test_str, 1)
    return result

def get_events(phrase):
    # if srl:
    #     results = predictor.predict(
    #         sentence=phrase
    #     )
    #     verbs = [v['verb'] for v in results['verbs'] if v not in ignore_verbs]
    #     print(results)
    #     return list(set(verbs))  #combine_contiguous_verbs(phrase, verbs)
    # else:
    doc = nlp(phrase)
    patterns =[[
                {'POS': 'AUX', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'}], # is driving 
               [
                {'POS': 'VERB', 'OP': '*'},
                {'POS': 'AUX', 'OP': '+'},
                {'POS':'PART', 'OP':'*'},
                {'POS': 'VERB', 'OP': '*'}], # did not
               
                [{'POS': 'VERB', 'OP': '+'},
                {'POS': 'ADP', 'OP': '*'},
                {'POS':'DET', 'OP':'*'},
                {'POS': 'PART', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'}], # did not drive
               
                [{'POS': 'VERB', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'}],
               
                [
                {'POS': 'PART', 'OP': '*'},
                {'POS': 'ADV', 'OP': '+'},
                {'POS': 'VERB', 'OP': '+'}],
                ]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add(key="Verb phrase", patterns=patterns)

    # call the matcher to find matches
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    verbs = filter_spans(spans)
    # print([(tkn.pos_, tkn.text) for tkn in doc])
    if not verbs:
        verbs = [tkn.text for tkn in doc if
                tkn.pos_ == "VERB" and tkn.pos_ != "is" and not tkn.is_stop and len(tkn.text) > 1]
    final_verbs = [str(v) for v in verbs] # combine_contiguous_verbs(phrase, verbs)

    return list(set(final_verbs))


def replace(p, subst, test_str):
    result = re.sub(p, subst, test_str, 1)
    return result


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = BATCH_SIZE
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
            ):

        with torch.no_grad():
            examples = queries
            decs = []
            i = 0
            for batch in list(chunks(examples, self.batch_size)):
                i += 1
                # print("Processing batch #", i)

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                if decode_method == "top-p":
                    summaries = self.model.generate(
                    do_sample=True,
                    # top_p = 0.98,
                    top_k = 1000,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_return_sequences=num_generate,
                    repetition_penalty = 1.5,
                    no_repeat_ngram_size = 2,
                    )
                else:
                    summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    repetition_penalty = 1.5,
                    no_repeat_ngram_size = 2,
                 
                    )
                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


def get_all_event_inferences(
    comet_model,
        sentence: str,
        rel: str,
        num_generate: int,
        decode_method: str,
):
    # get events
    events = get_events(sentence)
    heads = [replace(f"{events[i]}", f"<TGT> {events[i]} <TGT>", sentence).strip()
                          for i in range(len(events))]

    output = {}
    num_generate = max(1, num_generate)

    for i in range(len(events)):
      queries = []
      query = "{} {} [GEN]".format(heads[i], rel)
      queries.append(query)
      results = comet_model.generate(
          queries,
          decode_method=decode_method,
          num_generate=num_generate
      )
      output[events[i]] = results
    
   
    return output

if __name__ == "__main__":

    # sample usage
    models = ["sahithyaravi/comet-m", "sahithyaravi/comet-m-nli"] 
    models_init = []
    relations = ["xReason", "Causes", "HinderedBy", "isBefore", "isAfter", "HasPrerequisite"]
    context_inf_dict = defaultdict(lambda: defaultdict(list))
    for model in models:
        obj = Comet(model)
        obj.model.zero_grad()
        models_init.append(obj)

    sentences = ["My lawyer tells me you ’ Ve accepted our alimony proposal and the division of property , as well as the custody agreement - I keep the cat and you get the dog", "John insulted Mary, so she didn’t reply when he called her"]
    

    for i in range(len(models)): 
        model_obj = models_init[i]
        print(models[i])
        for sent in sentences:
            print(sent)
            for rel in relations:
                inferences = get_all_event_inferences(model_obj,sent, rel, num_generate=5, decode_method="beam")
                print(rel)
                print(inferences)

    


