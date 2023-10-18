from rouge_score import rouge_scorer, scoring
import pandas as pd
import argparse
import numpy as np
import json
import os
from collections import defaultdict
import random
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nltk import bleu
from nltk.translate import meteor
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction
from iteration_utilities import deepflatten
from evaluation.bert_score.bert_score import BertScore
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from comet_atomic2020_bart.utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
import nltk

BATCH_SIZE = 32

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

def comet_predict(input, model):
    comet = Comet(model)
    comet.model.zero_grad()
    print(input[0])
    L = comet.generate(input, decode_method="beam", num_generate=5)
    results = list(deepflatten(L, types=list)) 
    return list(chunks(results, 5))

def preprocess_inputs(args):
    with open(args.ground_truth_source_file, "r") as f:
        sources = f.readlines()
        sources = [line.replace("\n", "") for line in sources]
        
    with open(args.ground_truth_target_file, "r") as f:
        targets = f.readlines()
        targets = [line.replace("\n", "") for line in targets]

    generations = comet_predict(sources, args.comet_model)
    
    
  # Compute BLEU and ROUGE from the text predictions
    gold = defaultdict(list)
    predictions = defaultdict(set)

    for source, target, gens in tqdm(zip(sources, targets, generations)):
        curr_gold = target
        curr_preds = set([pred for pred in gens if len(pred) > 0])

        if len(curr_gold) > 0 and len(curr_preds) > 0:
            gold[source].append(curr_gold)
            predictions[source] = predictions[source].union(curr_preds)
    
    return gold, predictions

def measure_ngram_diversity(response_set):
    """
    Refer: https://github.com/GuyTevet/diversity-eval
    """
    def lines_to_ngrams(lines, n=3):
        ngrams = []
        for s in lines:
            words = [e for e in s.replace('.','').replace('\n','').split(' ') if e != '']
            ngrams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
        return ngrams

    def normalized_unique_ngrams(ngram_lists):
        """
        Calc the portion of unique n-grams out of all n-grams.
        :param ngram_lists: list of lists of ngrams
        :return: value in (0,1]
        """
        # print(ngram_lists)
        ngrams = [item for sublist in ngram_lists for item in sublist]  # flatten
        return len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.

    return normalized_unique_ngrams(lines_to_ngrams(response_set))


# def Similarity2DiversityMetric(response_set):
#     """
#     Refer: https://github.com/GuyTevet/diversity-eval
#     """
#     similarity_list = []
#     for i in range(len(response_set)):
#         for j in range(i):
#             similarity_list.append(self.similarity_metric(response_set[i], response_set[j]))
#     diversity_score = similarity2diversity_function(similarity_list)
#     return diversity_score


#     return diversity_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_source_file', type=str, default="training_data/multi-event/data-m/test.source",
    help='The ground truth .source file with <head> <relation> GEN')
    parser.add_argument('--ground_truth_target_file', type=str, default="training_data/multi-event/data-m/test.target",
    help='The ground truth .target file corresponding to ground_truth_source file')
    parser.add_argument('--comet_model', type=str, default="comet-atomic_2020_BART", #multi-event-m/best_tfmr
    help='The comet model you want to evaluate')
    parser.add_argument('--save_path', type=str, default="multi-event-m",
    help='where to save')
    args = parser.parse_args()
    # Get multiple targets per head 
    gold, predictions = preprocess_inputs(args)
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, rouge_scores, meteor_scores = [], [],[],[],[],[]
    smoothing = SmoothingFunction().method4
    rouge = Rouge()

    inputs = []
    generations = []
    references = []
    all_scores = []
    ngram_diversity_scores = []

    inf1 = defaultdict(list)
    inf2 = defaultdict(set)
    for input, curr_gold in gold.items():
        curr_predictions = list(predictions[input])

        # The refs and gold must be in the same size
        length = min(len(curr_gold), len(curr_predictions))

        if length > 0:
            hyps = curr_predictions
            refs = curr_gold

            # DIVERSITY
            ngram_diversity_scores.append(measure_ngram_diversity(hyps))

            # ROUGE
            scores = [np.max([rouge.get_scores(p, g)[0]["rouge-l"]["f"] for g in refs]) for p in hyps]
            sorted_scores = [s for s, x in sorted(zip(scores, hyps), reverse=True)][:length]
            sorted_hyps = [x for _, x in sorted(zip(scores, hyps), reverse=True)][:length]
            scores = sorted_scores
            hyps = sorted_hyps
            print(input, hyps, refs, scores)
            inputs.append(input)
            generations.append(hyps)
            references.append(refs)
            rouge_scores.extend(list(scores))
            all_scores.append(scores)
           
           # BLEU
            hyps = [tuple(h.split()) for h in hyps]
            refs = [tuple(r.split()) for r in refs]
            bleu1_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[1.0]) for pred in hyps])
            bleu2_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[0.5, 0.5]) for pred in hyps])
            bleu3_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[0.34, 0.33, 0.33]) for pred in hyps])
            bleu4_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[0.25, 0.25, 0.25, 0.25]) for pred in hyps])
            # meteor_scores.extend([meteor(refs, pred) for pred in hyps])

            # Top 2 Inferences for BERT score
            if len(hyps) > 1:
                inf1[input].append(hyps[0])
                inf2[input] = predictions[input].union(hyps[1])
    


    scorer = BertScore()
    sem_sim, scores = scorer.compute_score(gold, predictions)
    # sem_sim_self, scores = scorer.compute_score(inf1, inf2)

    bleu2 = 100.0 * np.mean(bleu2_scores)
    bleu4 = 100.0 * np.mean(bleu4_scores)
    rougel = 100.0 * np.mean(rouge_scores)
    ngram_diversity = 100.0 * np.mean(ngram_diversity_scores)

    print("\t".join([args.comet_model, f"Bleu-2: {bleu2:.3f}", f"Bleu-4: {bleu4:.3f}", f"Rouge-L: {rougel:.3f}", f"BertScore: {sem_sim:.3f}"]))
    
    with open(args.save_path + "/real_results.txt", "w") as f:
        f.write(args.comet_model + "\n")
        f.write( f"Rouge-L: {rougel:.3f}" + "\n")
        f.write(f"Bleu-2: {bleu2:.3f}"+ "\n")
        f.write(f"Bleu-4: {bleu4:.3f}" + "\n")
        f.write(f"BertScore: {sem_sim:.3f}" + "\n")
        f.write(f"Average DistinctNgrams: {ngram_diversity:.3f}" + "\n")
        # f.write(f"Average BERTScore_TopInferences: {sem_sim_self:.3f}" + "\n")


    df = pd.DataFrame()
    df["input"] = inputs
    df["generations"] = generations
    df["references"] = references
    df["scores"] = all_scores
    df.to_csv(args.save_path + "/final_results.csv")

if __name__ == "__main__":
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    result = scorer.score('The quick brown fox jumps over the lazy dog',
                        'The quick brown dog jumps on the log.')
    scores = {k: v.fmeasure for k, v in result.items()}
    scorer = BertScore()
    gold ={"xxx": ["bad", "good", "bad"]}
    predictions = {"xxx": ["bad", "good"]}
    score, scores = scorer.compute_score(gold, predictions)
    print(score)

    print("Diversity", measure_ngram_diversity(['The quick brown fox jumps over the lazy dog',
                        'The quick brown fox jumps over the lazy dog.',  'The quick brown fox jumps over the lazy dog.']))
    # print(scores)

    main()
   

