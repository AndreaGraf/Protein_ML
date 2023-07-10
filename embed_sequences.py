from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration
import torch
import h5py
import time
from typing import List, Tuple

#ensure that GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))


# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) 
    model = model.eval() 
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def get_embeddings( model, tokenizer, seqs:List[Tuple], per_residue:bool, per_protein:bool, max_length = 73, 
                   max_residues=4000, max_seq_len=1000, max_batch=100 ) -> dict:
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
                "residue_embs" : dict(), 
                 "protein_embs" : dict(),
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
        n_res_batch += seq_len #why?
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
                if batch_idx % 100 ==0:
                    print('processig batch ', batch_idx, f'satrting id: {pdb_ids[0]}' )
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx,:]
               
                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb[:s_len].mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results


def create_protein_batches(df:pd.DataFrame, protein_ids:list[str])->dict:
    """generates single protein input batches, that can be processed and saved in .h5 file format.
        Required because for processing large batches RAM limit is quickly exceeded.  

    Args:
        df (pd.DataFrame): input dataframe
        protein_ids (list[str]): list of protein name str

    Returns:
        dict: mapping protein names to sequence data
    """
    seq_lists ={}
    for pid in protein_ids:
        seq_dl = []
        #create unique id for each sequ and add to datalist  
        for i,r in df[df.pdbid == pid].iloc[:].iterrows(): 
            seq_dl.append((r.pdbid +'_'+ str(i), r.seq))
        seq_lists[pid] = seq_dl
    return seq_lists

 
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
  

def compute_embedding(model, tokenizer, seq_dict:dict, output_path:str) ->None:
    """compute embeddings for all proteins in seq_dict

    Args:
        model (_T5model_): pretrained model
        tokenizer (T5 Tokenizer): Tokenizer
        seq_dict (dict): input data. protein_id: seq list
        output_path (str): h5 file
    """
    for prot, seqs in seq_dict.items():
        print('processing protein:' , prot )
        results = get_embeddings(model.eval(), tokenizer, seqs=seqs, per_protein=True, per_residue=True, max_length=73)
    
        save_embeddings(os.path.join(output_path,"residue_embeddings1.h5"), results["residue_embs"])
        save_embeddings(os.path.join(output_path,"protein_embeddings1.h5"), results["protein_embs"])



### MAIN ###
def main()
    #preparing the input data for embedding
    data_dir = 'Data/data_sets/'
    filename = 'train_data_prot.csv'
    
    df = pd.read_csv(os.path.join(data_dir,filename), index_col=0)

    filename = 'val_data_prot.csv'
    df_v = pd.read_csv(os.path.join(data_dir,filename), index_col=0)

    filename = 'test_data_prot.csv'
    df_test = pd.read_csv(os.path.join(data_dir,filename), index_col=0)

    filename = 'test_proteins_data.csv'
    df_ho = pd.read_csv(os.path.join(data_dir,filename), index_col=0)

    #strip padding for embedding
    df['seq'] = df.seq.apply(lambda x: x.strip('-'))
    df_v['seq'] = df_v.seq.apply(lambda x: x.strip('-'))
    df_test['seq'] = df_test.seq.apply(lambda x: x.strip('-'))
    df_ho['seq'] = df_ho.seq.apply(lambda x: x.strip('-'))

    protein_ids = list(df.pdbid.unique())
    protein_v_ids = list(df_v.pdbid.unique())
    protein_test_ids = list(df_test.pdbid.unique())
    protein_ho_ids = list(df_ho.pdbid.unique())

    seq_lists = create_protein_batches(df, protein_ids)
    seq_lists1 = create_protein_batches(df_v, protein_v_ids)
    seq_lists2 = create_protein_batches(df_test, protein_test_ids)
    seq_lists3 = create_protein_batches(df_ho, protein_ho_ids)

    model, tokenizer = get_T5_model()
    print('model device:' ,model.device)

    #calculate the embeddings -- use GPU--
    compute_embedding(model.eval(),tokenizer, seq_lists, 'drive/MyDrive/protT5/output')

    wt_list = [(k, seq) for k, seq in wt_dict.items()]

    # get the embedding vectors
    wt_emb = get_embeddings(model.eval(), tokenizer, seqs=wt_list, per_protein=True, per_residue=True, max_length=73)
    save_embeddings('protT5/output/wt_embeddings.h5' , wt_emb['residue_embs'], 'a', )
    save_embeddings('protT5/output/wt_embeddings.h5' , wt_emb['residue_embs'], 'a', )



if __name__ == "__main__":
    main()