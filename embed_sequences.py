"""Script to generate embeddings for protein sequences using the T5 protein language model
https://github.com/agemagician/ProtTrans

Note: if single residues are embedded this generates very large files

"""
import sys
import os
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict



# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
def get_T5_model(device:str='cuda'):
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) 
    model = model.eval() 
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def get_embeddings( model, tokenizer, seqs:List[Tuple], per_residue:bool, per_protein:bool, word_size:int=1, max_length:int = 73, 
                   max_residues:int=4000, max_seq_len:int=1000, max_batch:int=100, max_entries:int=5000, h5_filename:str='residue_embeddings.h5',
                    h5_protein_filename:str ='protein_embeddings.h5', device:str='cuda' ) -> None:
    """  Generate embeddings via batch-processing

    Args:
        model: pretrained model
        tokenizer: T5model Tokeniser
        seqs (list): Tuples mapping of unique identifier to sequence
        per_residue (bool): indicates that embeddings for each residue in a protein should be returned.
        per_protein (bool): indicates that embeddings for a whole protein should be returned (average-pooling)
        max_length (int, optional): _description_. Defaults to 73.
        max_residues (int, optional): max_residues gives the upper limit of residues within one batch.
        max_seq_len (int, optional): maximum_sequence length. Defaults to 1000.
        max_batch (int, optional): _max_batch gives the upper number of sequences per batch. Defaults to 100.

    Returns:
        dict: embedding results
    """

    results = {
        "residue_embs": {},
        "protein_embs": {},
    }


    start = time.time()
    batch = []
    for seq_idx, (pdb_id, seq) in enumerate(seqs,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum(s_len for _, _, s_len in batch)
        n_res_batch += seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seqs) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = []

            # add_special_tokens adds extra token at the end of each sequence - this requires a max length of sequence length+1
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding='max_length', max_length= max_length)
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print(f"RuntimeError during embedding for {pdb_id} (L={seq_len})")
                continue

            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                
                    
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx,:]
               
                if per_residue: # store per-residue embeddings (Lx1024)
                    #average pooling over words
                    emb_s = emb.detach().cpu().numpy().squeeze()
                    emb_s =  np.array([emb_s[i:i+word_size].mean(axis=0) for i in range(0,emb_s.shape[0],word_size)])
                    results["residue_embs"][ identifier ] = emb
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb[:s_len].mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

        if len(results["residue_embs"]) >= max_entries or len(results["protein_embs"]) >= max_entries:
            print(f'Saving embeddings. last_id: {identifier}' )

            save_embeddings(h5_filename, results["residue_embs"])
            save_embeddings(h5_protein_filename, results["protein_embs"])

            # clean up dictionaries
            results = {
                "residue_embs": {},
                "protein_embs": {},
            }   


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    
    #save the remaining embeddings
    save_embeddings(h5_filename, results["residue_embs"])
    save_embeddings(h5_protein_filename, results["protein_embs"])


    print('\n############# EMBEDDING STATS #############')
    print(f'Total number of per-residue embeddings: {len(results["residue_embs"])}')
    print(f'Total number of per-protein embeddings: {len(results["protein_embs"])}')
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    


 
def save_embeddings(output_path:str, emb_dict:dict, saving_pattern = 'a')->None:
    """append results to .h5 lookup file, to free memory

    Args:
        output_path (str): h5 file
        emb_dict (dict): results from get_embeddings
        saving_pattern (str, optional): Defaults to 'a'.
    """
    with h5py.File(output_path, saving_pattern) as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)
  


### MAIN ###
def main():
    """
    The main function prepares the input data for embedding, 
    loads the T5 model, calculates the embeddings, and saves them to an output file.
    """
  
    #ensure that GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    emb_dir = os.path.join('Data', 'embeddings')


    args = sys.argv[1:]
    if len(args) == 0 or len(args) > 3:
        print ("invalid no of arguments given. Please specify the input file and optionally embeddings file.")
        print("Usage: python3 embed_sequences.py <sequences_csv_name> (<output_file>)")
        sys.exit(1)

    elif len(args) == 1:
        filename = args[0]
        embeddings_name = os.path.join(emb_dir,'embeddings.h5')
        protein_embeddings_name = os.path.join(emb_dir,'protein_embeddings.h5')
        wt_embeddings_name = os.path.join(emb_dir,'wt_embeddings.h5')
        wt_protein_embeddings_name = os.path.join(emb_dir,'wt_protein_embeddings.h5')

    elif len(args) == 2:
        filename = args[0]
        embeddings_name = args[1]
        directory, emb_filename = os.path.split(embeddings_name)
        protein_embeddings_name = os.path.join(directory, 'protein_'+emb_filename)
        wt_embeddings_name = os.path.join(directory, 'wt_'+emb_filename)
        wt_protein_embeddings_name = os.path.join(directory, 'wt_protein_'+emb_filename)

    #load data
    df = pd.read_csv(filename, index_col=0)

    #strip padding for embedding
    df['seq'] = df.seq.apply(lambda x: x.strip('-'))

    #write only indices after last saved embedding
    df = df[df.index > 130245]

    #get lists of sequences and wild types for embedding. 
    # Use WT_name as identifier for wt embedding
    seq_lists = list(df[['id', 'seq']].itertuples(index=False, name=None))
    wt_lists = list(df.drop_duplicates(subset=['WT_name'])[['WT_name', 'wt_p']].itertuples(index=False, name=None))


    model, tokenizer = get_T5_model(device=device)
    print('model device:', model.device)

    #calculate the embeddings -- use GPU if available
    get_embeddings(model, tokenizer, seqs=seq_lists, per_protein=True, per_residue=True, max_length=73, max_batch=5000,
                    h5_filename=embeddings_name, h5_protein_filename=protein_embeddings_name, device=device)

    
    get_embeddings(model, tokenizer, seqs=wt_lists, per_protein=True, per_residue=True, max_length=73, max_batch=5000,
                    h5_filename=wt_embeddings_name, h5_protein_filename=wt_protein_embeddings_name, device=device)



    #

    # get the embedding vectors
    #wt_emb = get_embeddings(model.eval(), tokenizer, seqs=wt_list, per_protein=True, per_residue=True, max_length=73)
    #save_embeddings('protT5/output/wt_embeddings.h5' , wt_emb['residue_embs'], 'a', )
    #save_embeddings('protT5/output/wt_embeddings.h5' , wt_emb['residue_embs'], 'a', )



if __name__ == "__main__":
    main()