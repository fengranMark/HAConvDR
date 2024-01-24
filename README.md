# HAConvDR

A temporary repository of our ARR submission - History-Aware Conversational Dense Retrieval.

# Environment Dependency

Main packages:
- python 3.8
- torch 1.8.1
- transformer 4.2.0
- numpy 1.22
- faiss-gpu 1.7.2
- pyserini 0.16

# Running Steps

## 1. Download data and Preprocessing

Four public datasets can be downloaded from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa), and [TREC-CAST](https://www.treccast.ai/). Data preprocessing can refer to "preprocess" folder. The file with "PRJ" prefix is to process data for generating pseudo relevant judgment.

## 2. Generate pseudo relevant judgment

First, generate the qrel file by function "create_label_rel_turn" in "PRJ_topiocqa.py" and "PRJ_qrecc.py"

Second, generate the pseudo relevant judgment (PRJ) by
```
python test_PRJ_topiocqa.py --config=Config/test_PRJ_topiocqa.toml
python test_PRJ_qrecc.py --config=Config/test_PRJ_qrecc.toml
```

The output file "train_rel_label.json" contains the PRJ for each turn.

## 3. Retrieval Indexing (Dense and Sparse)

To evaluate the trained model by HAConvDR, we should first establish index for both dense and sparse retrievers. The sparse retrieval is used for generating bm25-hard negatives.

### 3.1 Dense
For dense retrieval, we use the pre-trained ad-hoc search model ANCE to generate passage embeddings. Two scripts for each dataset are provided by running:

    python gen_tokenized_doc.py --config=Config/gen_tokenized_doc.toml
    python gen_doc_embeddings.py --config=Config/gen_doc_embeddings.toml

### 3.2 Sparse

For sparse retrieval, we first run the format conversion script as:

    python convert_to_pyserini_format.py
    
Then create the index for the collection by running

    bash create_index.sh
    
Finally, the produced bm25 rank-list can be used to generate bm25-hard negatives by "extract_doc_content_of_bm25_hard_negs_for_train_file" function in preprocess files.

## 4. Train HAConvDR

To train HAConvDR, please run the following commands. The pre-trained language model we use for dense retrieval is [ANCE](https://github.com/microsoft/ANCE).

    python train_convretriever_topiocqa.py --pretrained_encoder_path="checkpoints/ad-hoc-ance-msmarco" \ 
      --train_file_path=$train_file_path \ 
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \ 
      --per_gpu_train_batch_size=64 \ 
      --num_train_epochs=10 \
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=64 \
      --max_concat_length=512 \ 
      --use_PRL=True \
      --is_train=True \
      --is_prepos_neg=True \
      --is_pseudo_prepos=True \
      --is_PRF=False \ % True for PRF setting
      --alpha=1
      
## 5. Retrieval evaluation

Now, we can perform retrieval to evaluate the ConvHACL-trained dense retriever by running:

    python test_retrieval_topiocqa.py --pretrained_encoder_path=$trained_model_path \ 
      --passage_embeddings_dir_path=$passage_embeddings_dir_path \ 
      --passage_offset2pid_path=$passage_offset2pid_path \
      --qrel_output_path=$qrel_output_path \ % output dir
      --output_trec_file=$output_trec_file \
      --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \ % gold qrel file
      --per_gpu_train_batch_size=4 \ 
      --test_type=convqp \ % convqa for qrecc
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=64 \
      --max_concat_length=512 \ 
      --use_PRL=False \
      --is_PRF=False \
      --is_train=False \
      --top_k=100 \
      --rel_threshold=1 \
      --passage_block_num=$passage_block_num \
      --use_gpu=True
