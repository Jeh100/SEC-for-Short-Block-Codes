from sionna.phy.fec.utils import load_alist, alist2mat
import os,csv, numpy as np, tensorflow as tf, json
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import Mapper, Demapper, Constellation
from sionna.phy.fec.linear.encoding import LinearEncoder
from sionna.phy.fec.linear.decoding import OSDecoder
from evaluate import load
from utils import denoise_text,systematic
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR') 

physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        tf.config.set_visible_devices(physical_gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
    except RuntimeError as e:
        print(e)


src="snli_eval.ref"
bleu_metric = load('sacrebleu')
rouge_metric = load('rouge')

with open(src, newline='', encoding='utf-8') as file:
    reader = list(csv.reader(file))
    dataset_size = len(reader)

num_bits_per_symbol = 1 # BPSK
batch_size = 1

awgn_channel = AWGN() 
al = load_alist(path="64_128.alist")
pcm, k, n, code_rate = alist2mat(al)
G_sys,_ = systematic(pcm)


encoder = LinearEncoder(enc_mat=G_sys, is_pcm=False)
decoder = OSDecoder(G_sys, t=4, is_pcm=False)
mapper = Mapper("pam", num_bits_per_symbol)
constellation = Constellation("pam", num_bits_per_symbol, trainable=False)
demapper = Demapper("app", constellation=constellation)
save_path = "bart_cor_v1"
model = ORTModelForSeq2SeqLM.from_pretrained(save_path, export=True,provider="CUDAExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(save_path)
sentence_index = 0 

results = []
counter = 0
ebno = np.arange(1.5, 4, 0.5)
error_block_targets = [2000, 1500, 1000, 1000, 500, 300, 100, 30]


for ebno_db, threshold in zip(ebno, error_block_targets):
    block_num = 0
    ori_error_blocks = 0
    cle_error_blocks = 0
    ori_BLER = 0
    cle_BLER = 0
    bleu_cleaned = 0
    rouge_cleaned = 0
    bleu_decoded = 0
    rouge_decoded = 0
    dur_time = 0
    while ori_error_blocks < threshold:
        test_instance = reader[sentence_index % dataset_size][0]   
        sentence_index += 1
        encoded = [ord(char) for char in test_instance]
        bin_encoded = [format(value, '08b') for value in encoded]
        bits_list = [int(bit) for binary in bin_encoded for bit in binary]

        segments = [bits_list[i:i+k] for i in range(0, len(bits_list), k)]
        tf.random.set_seed(42)
        recovered_bits = []
        no = ebnodb2no(ebno_db, num_bits_per_symbol, code_rate)
        for segment in segments:
            if len(segment) < k:
                segment = segment + [0] * (k - len(segment))
            b = tf.constant(segment, dtype=tf.float32)
            b = tf.reshape(b, [batch_size, k])
            c = encoder(b)
            x = mapper(c)
            y = awgn_channel(x, no) 
            llr = demapper(y, no)  
            b_hat  = decoder(llr)
            b_hat = tf.round(b_hat[:, :k])
            arr = b_hat.numpy().flatten()
            b_str_for_dec = ''.join(str(int(val)) for val in arr) 
            recovered_bits.extend(b_str_for_dec)
        recovered_bits = recovered_bits[:len(bits_list)]


        binary_strings = [''.join(map(str, recovered_bits[i:i+8])) for i in range(0, len(recovered_bits), 8)]
        int_decoded = [int(binary, 2) for binary in binary_strings]
        decoded = ''.join(chr(value) if 32 <= value <= 126 else '?' for value in int_decoded) 
        cleaned_sentence = denoise_text(model,tokenizer,decoded)
        

        bleu_decode = bleu_metric.compute(
            predictions=[decoded], references=[[test_instance]])
        rouge_decode = rouge_metric.compute(
            predictions=[decoded], references=[[test_instance]])
        bleu_cleane = bleu_metric.compute(
            predictions=[cleaned_sentence], references=[[test_instance]])
        rouge_cleane = rouge_metric.compute(
            predictions=[cleaned_sentence], references=[[test_instance]])
        
        bleu_decoded = bleu_decoded + bleu_decode["score"]
        rouge_decoded = rouge_decoded + (rouge_decode["rougeL"] * 100.0)
        bleu_cleaned = bleu_cleaned + bleu_cleane["score"]
        rouge_cleaned = rouge_cleaned + (rouge_cleane["rougeL"] * 100.0)
        block_num += 1 
        if decoded != test_instance:
            ori_error_blocks += 1  
        if cleaned_sentence != test_instance:
            cle_error_blocks += 1
    ori_BLER = ori_error_blocks/block_num
    cle_BLER = cle_error_blocks/block_num
    bleu_dec = bleu_decoded / block_num
    rouge_dec = rouge_decoded / block_num
    bleu_cle = bleu_cleaned / block_num
    rouge_cle = rouge_cleaned / block_num
    print(f"Ebno:{ebno_db} original BLER={ori_BLER} BLEU={bleu_dec} ROUGE={rouge_dec} block_number={block_num}")
    results.append({
    "Ebno":ebno_db,
    "ori BLER": ori_BLER,
    "BLEU": bleu_dec,
    "ROUGE": rouge_dec,
    "block_number": block_num
    })
    print(f"Ebno:{ebno_db} cleaned BLER={cle_BLER} BLEU={bleu_cle} ROUGE={rouge_cle} block_number={block_num}")
    results.append({
    "Ebno":ebno_db,
    "cleaned BLER": cle_BLER,
    "BLEU": bleu_cle,
    "ROUGE": rouge_cle,
    "block_number": block_num
    })
    with open('performance_short_new.jsonl', 'a', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
        results = []
    print("----------------------------------------")
    


