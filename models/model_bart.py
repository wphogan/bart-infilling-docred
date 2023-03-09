import torch.cuda
from transformers import BartForConditionalGeneration, BartTokenizer


class Bart:
    def __init__(self, debug=False):
        # Load model and tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_size = 'large' if self.device == 'cuda' else 'base'
        self.debug = debug
        if self.debug:
            self.model_size = 'base'
            self.printd('--> TEMP BASE SIZE FOR DEBUG', highlight=True)
        self.model_name = f'facebook/bart-{self.model_size}'
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name, forced_bos_token_id=0)
        self.model = self.model.to(self.device)
        self.tok = BartTokenizer.from_pretrained(self.model_name)
        self.printd(f'--> {self.device}, {self.model_name}', highlight=True)

    def printd(self, print_str, highlight=False):
        if highlight: print('x' * 30)
        if self.debug: print(print_str)
        if highlight: print('x' * 30)

    def gen(self, input_str, orig_str, masked_sent_list, replaced_txt_no_ws_str, mask_idx, first_word_afer_mask):
        input_ids = self.tok([input_str], return_tensors="pt")["input_ids"].to(self.device)
        generated_ids = self.model.generate(input_ids, do_sample=True, max_new_tokens=512, num_return_sequences=5,
                                            temperature=1.2, repetition_penalty=1.5)
        self.printd(f'GEN IDS SIZE: {generated_ids.size()}')
        out = self.tok.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.printd(f'REPLACED TXT:\n {replaced_txt_no_ws_str}')
        self.printd(f'FIRST WORD AFTER MASK:\n {first_word_afer_mask}')
        self.printd(f'ORIG:\n {orig_str}')
        self.printd(f'IN:\n {input_str}')
        self.printd('OUT:')
        n_len_best = 0
        best_candidate = {
            'sent': False,
            'len_infill': -1,
            'end_idx': -1
        }
        for i, o in enumerate(out):
            self.printd(f'{i}, {o}')
            o_list = o.split()
            try:
                end_idx = o_list.index(first_word_afer_mask, mask_idx)
            except ValueError:  # skip infills that modify words beyond the mask
                continue
            o_infilled = ''.join(o_list[mask_idx:end_idx])
            # Check for two conditions:
            # 1. infilled span is different than original span
            # 2. infilled span does not end in a period
            if o_infilled != replaced_txt_no_ws_str and '.' not in o_infilled:
                if n_len_best < len(o_infilled):
                    n_len_best = len(o_infilled)
                    best_candidate['sent'] = masked_sent_list[:mask_idx] + o_list[mask_idx:end_idx] + masked_sent_list[
                                                                                                      mask_idx + 1:]
                    best_candidate['len_infill'] = len(o_list[mask_idx:end_idx])
                    best_candidate['end_idx'] = end_idx
                    self.printd('FOUND BEST CANDIATE!')
            self.printd(f'({mask_idx}, {end_idx}): {o_infilled}')

        if self.debug:
            x = input('continue?...')
        return best_candidate
