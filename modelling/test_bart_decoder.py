#import bartdecoder and bart config
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
import torch


decoder = BartDecoder(BartConfig())
input = torch.rand(32,1024).to(torch.int64)
output = decoder(input)
for out in output:
    print(out.shape)