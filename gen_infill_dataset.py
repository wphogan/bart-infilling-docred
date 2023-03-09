import argparse
import json
import os
from os.path import join

from tqdm import tqdm

# Local imports
from models.model_bart import Bart
from utils.utils import load_docred_json, update_vertex_set, gen_new_sents


def main(args):
    ######## Run settings ########
    file_input = 'data/train_annotated_graph_replacement.json'
    file_output = 'output/train_annotated_infilled_graph_replacement_docred.json'

    # Load data
    data = load_docred_json(join('data', file_input))
    all_new_data = []  # Holds new dataset

    # Load model
    model = Bart(debug=args.debug)
    global_unreplaced_sent_count = 0
    global_replaced_sent_count = 0

    print(f'Processing {len(data)} documents from {file_input}. Saving to {file_output}')
    x = input('Press enter to continue...')

    # Iterate through all documents
    for i in tqdm(range(len(data))):
        row = data[i]
        vertex_set = row['vertexSet']
        sents = row['sents']

        # 1. Generate new setences
        doc_new_sents, doc_new_sents_debug, doc_new_sents_mask_end_idx, n_unreplaced = gen_new_sents(model, vertex_set,
                                                                                                     sents)
        global_unreplaced_sent_count += n_unreplaced
        global_replaced_sent_count += len(doc_new_sents) - n_unreplaced

        # 2. Update spans in vertex set
        new_vertex_set = update_vertex_set(vertex_set, doc_new_sents, doc_new_sents_debug, doc_new_sents_mask_end_idx,
                                           is_debug=args.debug)

        # 3. Append updated doc to list of new data
        replacement_sents = [sent for sent_id, sent in doc_new_sents.items()]
        assert len(sents) == len(replacement_sents), 'must have same in/out sent count'

        row['sents'] = replacement_sents
        row['vertexSet'] = new_vertex_set

        all_new_data.append(row)
        # END DOC LOOP

    # Write new data file
    with open(join('output', file_output), 'w') as f:
        f.write(json.dumps(all_new_data))
    print(f'--> Wrote file: {file_output}')

    print('global_unreplaced_sent_count: ', global_unreplaced_sent_count)
    print('global_replaced_sent_count: ', global_replaced_sent_count)
    print('total sents in dataset: ', global_replaced_sent_count + global_unreplaced_sent_count)


if __name__ == '__main__':
    print('(CLEANED) Starting file: ', os.path.basename(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, required=False)
    args = parser.parse_args()
    main(args)
    print('\nCompelted file: ', os.path.basename(__file__))
