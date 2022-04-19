from transformers import T5TokenizerFast
import torch
import os
from pathlib import Path


ROOT = Path(os.path.realpath(os.path.expanduser(__file__))).parents[0]
SESS = torch.load(str(ROOT /"t5-summarizer.pkl"))
TKNR = T5TokenizerFast.from_pretrained('t5-small')

class Model(object):

    model = SESS
    tokenizer = TKNR

    @staticmethod
    def metadata():
        return {
            'signature_name': 'serving_default',
            'inputs':{
                'text_data': {
                    'dtype':'str'
                }
            },
            'outputs':{
                'summarized_text':{
                    'dtype':'str'
                }
            }
        }


    #helper funcs:
    def __tokenize(cls,text):
        enc_text = cls.tokenizer(
            "summarize:"+text+cls.tokenizer.eos_token,
            max_length = 512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        return enc_text['input_ids'], enc_text['attention_mask']
        

    def __sum_gen(cls, in_ids, attn_mask):
        generated_ids = cls.model.generate(
            input_ids = in_ids,
            attention_mask=attn_mask,
            max_length=128,
            num_beams=4,
            repetition_penalty = 4.5,
            length_penalty=1.0,
            early_stopping=True
        )
        return generated_ids   

    
    def __decode(cls, sequence):
        pred_sum = [
            cls.tokenizer.decode(gen_id, skip_special_tokens=True, cean_up_tokenization_spaces=True)
            for gen_id in sequence
        ]
        return "".join(pred_sum)


    @classmethod
    def summarize(cls,data):
        enc_in, enc_mask = cls.__tokenize(cls,data)
        enc_sum = cls.__sum_gen(cls,enc_in, enc_mask)
        output = cls.__decode(cls,enc_sum)
        return output
