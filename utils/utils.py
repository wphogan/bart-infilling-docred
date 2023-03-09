import json
import string
import itertools
import jellyfish
from collections import defaultdict
from random import choice
from rapidfuzz import fuzz


def gen_new_sents(model, vertex_set, sents):
    doc_new_sents = {}
    doc_new_sents_debug = {}
    doc_new_sents_mask_end_idx = {}
    doc_n_unreplaced_sents = 0
    # a. Find all entity locations
    # Do not mask existing entities
    doc_ent_idxs = defaultdict(set)
    for v in vertex_set:
        for d in v:
            span_start = d['pos'][0]
            span_end = d['pos'][1]
            ent_idxs = set([x for x in range(span_start, span_end)])
            doc_ent_idxs[d['sent_id']].update(ent_idxs)

    # b. Insert mask token and infill to gen new sents
    for sent_id, current_sentence in enumerate(sents):
        ent_idxs = doc_ent_idxs[sent_id]
        current_sentence = sents[sent_id]
        current_sentence_str = ' '.join(current_sentence)
        current_sentence_list = docred_sent_split(current_sentence_str)
        if current_sentence_list != current_sentence:
            print('WARNING: discrepancy with sentence split -->')
            print('Orig: ', current_sentence)
            print('Split+combined: ', current_sentence_list)

        # Default vals for debugging only
        mask_idx = -1
        replaced_text_no_ws = ''
        first_word_afer_mask = None

        # Loop until a good new sent is generated
        new_sent_dict = {
            'sent': False,
            'len_infill': -1,
            'end_idx': -1
        }
        max_iters = 10
        while not new_sent_dict['sent']:
            mask_idx, adder = gen_rand_mask_idx(current_sentence, ent_idxs)
            if mask_idx < 0 or max_iters == 0:
                # No non-entity ids available for masking
                doc_n_unreplaced_sents += 1
                new_sent_dict['sent'] = current_sentence.copy()
                break
            temp_sent = current_sentence.copy()
            masked_sent_list = temp_sent[0:mask_idx] + ['<mask>'] + temp_sent[mask_idx + adder:]
            first_word_afer_mask = temp_sent[mask_idx + adder]
            replaced_text_no_ws = ''.join(temp_sent[mask_idx:mask_idx + adder])
            assert len(masked_sent_list) + adder - 1 == len(current_sentence)
            masked_sent_str = ' '.join(masked_sent_list)
            new_sent_dict = model.gen(masked_sent_str, current_sentence_str, masked_sent_list, replaced_text_no_ws,
                                      mask_idx,
                                      first_word_afer_mask)
            max_iters -= 1

        doc_new_sents[sent_id] = new_sent_dict['sent']
        doc_new_sents_debug[sent_id] = (
            mask_idx, replaced_text_no_ws, first_word_afer_mask, current_sentence_str, new_sent_dict['sent'])
        doc_new_sents_mask_end_idx[sent_id] = (mask_idx, adder, new_sent_dict['end_idx'], new_sent_dict['len_infill'])

    doc_new_sents = {k: v for k, v in sorted(list(doc_new_sents.items()))}  # sort on sent id
    return doc_new_sents, doc_new_sents_debug, doc_new_sents_mask_end_idx, doc_n_unreplaced_sents


def is_match(ent_name, span_list):
    ent_name_list = ent_name.split()
    while '\xa0' in span_list: span_list.remove('\xa0')
    while '\xa0' in ent_name_list: ent_name_list.remove('\xa0')
    no_ws_ent_name = ''.join(ent_name_list)
    no_ws_span_str = ''.join(span_list)
    match = (no_ws_span_str == no_ws_ent_name)
    if match:
        return True
    else:
        score = fuzz.ratio(no_ws_span_str, no_ws_ent_name)
        if score > 95.0:
            # print("MATCH!")
            return True
    return False


def update_vertex_set(vertex_set, doc_new_sents, doc_new_sents_debug, doc_new_sents_mask_end_idx, is_debug):
    new_vertex_set = []
    ent_name_counter = defaultdict(int)
    for v in vertex_set:
        entries = []
        for d in v:
            sent_id = d['sent_id']
            ent_name = d['name']
            if '\n' in ent_name:
                # fix for '4.\nStranmillis Road' ent name in train_annotated data
                split = ent_name.split('\n')[1]
                ent_name = split
                d['name'] = split
            span_start_old = d['pos'][0]
            span_end_old = d['pos'][1]

            target_sent = doc_new_sents[sent_id]
            mask_idx, orig_addr, end_idx, len_infill = doc_new_sents_mask_end_idx[sent_id]
            span_list = target_sent[span_start_old:span_end_old]
            span_len = span_end_old - span_start_old
            new_adder = len_infill - orig_addr
            if is_debug:
                print('mask_idx: ', mask_idx)
                print('orig_addr: ', orig_addr)
                print('end_idx: ', end_idx)
                print('len_infill: ', len_infill)
                print('mask_idx + len_infill: ', mask_idx + len_infill)
                print('equal to end idx: ', (mask_idx + len_infill == end_idx))
                print('span_start_old: ', span_start_old)
                print('span_end_old: ', span_end_old)
                print('target_sent: ', target_sent)
                print('ent_name: ', ent_name)
                print('ent_span_str_old: ', span_list)

            if span_start_old >= mask_idx and span_start_old <= end_idx:
                if len_infill - orig_addr == 0:  # next ent should occur right after infilled text
                    span_start_new = end_idx
                else:  # a word or two after infilled text before next ent
                    span_start_new = span_start_old + len_infill - orig_addr
                span_end_new = span_start_new + span_len
                span_list = target_sent[span_start_new:span_end_new]
                if is_debug:
                    print('span_start_new: ', span_start_new)
                    print('span_end_new: ', span_end_new)
                    print('ent_span_str_new: ', span_list)
                    print('case: old span within in infilled text')
                d['pos'] = [span_start_new, span_end_new]

            elif span_start_old > end_idx:
                adder = new_adder
                span_start_new = span_start_old + adder
                span_end_new = span_start_new + span_len
                span_list = target_sent[span_start_new:span_end_new]
                if is_debug:
                    print('new adder: ', adder)
                    print('span_start_new: ', span_start_new)
                    print('span_end_new: ', span_end_new)
                    print('ent_span_str_new: ', span_list)
                    print('case: span exists after mask')
                d['pos'] = [span_start_new, span_end_new]

            match = is_match(ent_name, span_list)
            if not match:
                # x = input('About to remove NULL spaces and search. Pause...')
                ent_name_list = ent_name.split()
                try:
                    span_start_new = target_sent.index(ent_name_list[0], span_start_old - 1)
                except ValueError:
                    print('--> not found in original sent: ', ent_name_list[0])
                    try:
                        trim = ent_name_list[0][:-1]
                        span_start_new = target_sent.index(trim)
                    except ValueError:
                        print('--> OH NO, trimmed version still not found in original sent: ')
                        try:
                            span_start_new = target_sent.index(ent_name_list[0])
                        except ValueError:
                            span_start_new = 0
                span_end_new = span_start_new + span_len
                span_list = target_sent[span_start_new:span_end_new]
                match = is_match(ent_name, span_list)
                if is_debug:
                    print('NOT FOUND! Last resort search:')
                    print('ent_span_str_new: ', span_list)
                    print('FINAL MATCH: ', match)
                d['pos'] = [span_start_new, span_end_new]
                # x = input('...')

            if is_debug:
                print()
                print('x' * 30)
                print()

            entries.append(d)
        new_vertex_set.append(entries)
    return new_vertex_set


def find_new_span(target_sent, ent_name, nth_ent=1):
    # Name -> loc in sentence
    # Case 1: 'Microsoft' -> 'Microsoft'
    # Case 2: 'Asian Air, Inc.' --> 'Asian', 'Air', ',', 'Inc.'
    # Case 3: 'Microsoft' -> 'Microsoft'...and then another...'Microsoft'
    ent_name_split = ent_name.split(' ')

    first_word = ent_name_split[0]
    if ent_name_split[0][-1] in string.punctuation and ent_name_split[0][-1] != '.':
        first_word = ent_name_split[0][:-1]

    for i, word_or_punct in enumerate(target_sent):
        if word_or_punct == first_word:
            found, nth_ent = substring_finder(i, target_sent, ent_name_split, nth_ent)
            if found:
                return found
        # Check for plural version of word
        if word_or_punct + 's' == first_word:
            word_or_punct += 's'
            found, nth_ent = substring_finder(i, target_sent, ent_name_split, nth_ent)
            if found:
                return found
    return False, False


def substring_finder(i, target_sent, ent_name_split, nth_ent):
    words_in_span = target_sent[i:i + len(ent_name_split)]
    adder = 0
    for w in words_in_span:
        if w in string.punctuation and w != '-':
            adder += 1
    updated_words_in_span = target_sent[i:i + len(ent_name_split) + adder]
    combine_words = []
    for w in updated_words_in_span:
        if w not in string.punctuation and w != '-':
            combine_words.append(w)
        else:
            combine_words[-1] = combine_words[-1] + w

    x = ' '.join(combine_words)
    y = ' '.join(ent_name_split)
    x_ext = ''
    try:
        x_ext = combine_words + target_sent[i + len(ent_name_split) + adder + 1]
        x_ext = ' '.join(x_ext)
    except:
        pass
    dist = jellyfish.levenshtein_distance(x, y)
    dist_ext = jellyfish.levenshtein_distance(x_ext, y)

    if dist / len(y) <= len(ent_name_split) / len(y):  # allow one 's' per word
        nth_ent -= 1
        if nth_ent == 0:
            return (i, i + len(ent_name_split) + adder), nth_ent
        else:
            return False, nth_ent
    elif dist_ext / len(y) <= len(ent_name_split) / len(y):  # allow one 's' per word
        nth_ent -= 1
        if nth_ent == 0:
            return (i, i + len(ent_name_split) + adder), nth_ent
        else:
            return False, nth_ent
    return False, nth_ent


def debug_missing_ent(doc_new_sents_debug, sent_id, target_sent, ent_name, ent_name_counter, ent_name_sent_id):
    mask_idx, replaced_text_no_ws, first_word_afer_mask, current_sentence_str, new_sent = doc_new_sents_debug[sent_id]
    # x = input('paused. error found. enter to show error.')
    print('target_sent', target_sent)
    print('ent_name: ', ent_name)
    print('ent_name_counter[ent_name_sent_id]: ', ent_name_counter[ent_name_sent_id])
    print('mask:', mask_idx)
    print('replaced_text_no_ws:', replaced_text_no_ws)
    print('first_word_afer_mask:', first_word_afer_mask)
    print('orig sent:', current_sentence_str)
    print('new_sent:', new_sent)
    # x = input('cont?...')


def load_docred_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def docred_sent_split(sent):
    return sent.split(' ')


def gen_rand_mask_idx(sent, ent_idxs):
    ent_idxs.add(len(sent) - 1)  # make sure last word in sent is not masked id
    ent_idxs.add(0)  # make sure first word is not masked id
    len_sent = len(sent)
    options = tuple([x for x in range(0, len_sent) if x not in ent_idxs])

    at_least_one_nonpunt_word = False
    for idx_option in options:
        if at_least_one_nonpunt_word:
            continue
        if sent[idx_option] not in string.punctuation:
            at_least_one_nonpunt_word = True

    if not at_least_one_nonpunt_word or len(options) == 0:
        # print(f'Found sentence with no mask options. Skipping.')
        return -1, -1

    mask_idx = choice(options)
    while sent[mask_idx] in string.punctuation:
        mask_idx = choice(options)

    # mask expander
    adder = 1
    while mask_idx + adder in options:
        adder += 1
    adder = min(adder, 5)
    return mask_idx, adder


def count_continuous_missing_words(trimmed_out_list, next_word_after_mask, crts, ins, otlst):
    count = 0
    for i, word in enumerate(trimmed_out_list):
        if next_word_after_mask in word:
            return count
        else:
            count += 1
    return False


def align_two_lists(list1, list2, missing="MISSING"):
    value = list(zip(*list(align_iterables([list1, list2], missing=missing))))
    if not value:
        return [[], []]
    else:
        a, b = value
        return [list(a), list(b)]


def align_iterables(inputs, missing=None):
    """Align sorted iterables

    Yields tuples with values from the respective `inputs`, placing
    `missing` if the value does not exist in the corresponding
    iterable.

    Example: align_generator('bc', 'bf', '', 'abf') yields:
        (None, None, None, 'a')
        ('b', 'b', None, 'b')
        ('c', None, None, None)
        (None, 'f', None, 'f')
    """
    End = object()
    iterators = [itertools.chain(i, [End]) for i in inputs]
    values = [next(i) for i in iterators]
    while not all(v is End for v in values):
        smallest = min(v for v in values if v is not End)
        yield tuple(v if v == smallest else missing for v in values)
        values = [next(i) if v == smallest else v
                  for i, v in zip(iterators, values)]
