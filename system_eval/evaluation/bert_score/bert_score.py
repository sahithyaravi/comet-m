from evaluation.bert_score.score import score
# Code for BertScore reused from original implementation: https://github.com/Tiiiger/bert_score

class BertScore:
    def __init__(self):
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        hyp_input = []
        ref_input = []
        same_indices = []
        for id in imgIds:
            hypo =  list(res[id])
            ref = gts[id]

            # Sanity check.
            # assert(type(hypo) is list) prev
            # assert(len(hypo) == 1) prev
            assert(type(ref) is list)
            assert(len(ref) >= 1)
            for hyp in hypo:
                hyp_input += [hyp] * len(ref)
                ref_input += ref

            # hyp_input += [hypo[0]] * len(ref) prev
            # ref_input += ref
            same_indices.append(len(ref_input))

        p, r, f_scores = score(hyp_input, ref_input)
 
        prev_idx = 0
        aggreg_f1_scores = []
        for idx in same_indices:
            aggreg_f1_scores.append(f_scores[prev_idx: idx].mean().cpu().item())
            prev_idx = idx

        return sum(aggreg_f1_scores)/len(aggreg_f1_scores), aggreg_f1_scores

    def method(self):
        return "Bert Score"
