"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : test_sft_context_and_normalization.py

Context extension and response-balanced losses must preserve their contracts."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader,TensorDataset

from model.gpt_model import GPT
from training.loss import autoregressive_cross_entropy,repetition_unlikelihood,training_loss_with_unlikelihood
from training.train import TrainConfig,train_model
from tools.hf_sft_experiment import CachedEncoder
from tokenizer.tokenizer import Tokenizer


class ContextAndNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_context_extension_preserves_short_logits_and_updates_all_masks(self):
        torch.manual_seed(11)
        for flash in (True,False):
            model=GPT(16,8,embedding_dim=8,num_heads=2,num_blocks=1,dropout=0).eval()
            model.blocks[0].attention._use_flash=flash
            x=torch.randint(0,16,(2,8));before=model(x).detach()
            old=model.embedding.position_emb.weight.detach().clone()
            model.resize_context(16)
            torch.testing.assert_close(model(x),before,atol=0,rtol=0)
            torch.testing.assert_close(model.embedding.position_emb.weight[:8],old,atol=0,rtol=0)
            self.assertEqual(tuple(model.blocks[0].attention.causal_mask.shape),(1,1,16,16))
            self.assertIs(model.output_head.weight,model.embedding.token_emb.weight)
            model(torch.randint(0,16,(2,16))).sum().backward()
            self.assertGreater(model.embedding.position_emb.weight.grad[8:].abs().sum().item(),0)
            with self.assertRaises(ValueError):model.resize_context(4)

    def test_extended_context_artifact_roundtrips(self):
        import training.train as training
        from inference.generate import load_model_from_final_state
        model=GPT(16,8,embedding_dim=8,num_heads=2,num_blocks=1,dropout=0).eval()
        model.resize_context(16)
        x=torch.randint(0,16,(1,16))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sft.pt'
            training.save_final_model_state(model,path,torch.device('cpu'),TrainConfig())
            restored,_=load_model_from_final_state(path,None,'cpu')
            self.assertEqual(restored.seq_length,16)
            torch.testing.assert_close(restored(x),model(x),atol=0,rtol=0)

    def test_sequence_losses_equal_independent_example_means(self):
        torch.manual_seed(12)
        logits=torch.randn(3,8,6,dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[1,2,3,1,2,4,-100,-100],[1,2,-100,-100,-100,-100,-100,-100],[2,1,2,3,1,2,4,5]])
        for function,kwargs in ((autoregressive_cross_entropy,{'label_smoothing':.1,'z_loss_weight':.01}),
                                (repetition_unlikelihood,{'ngram_size':3,'window':8})):
            actual=function(logits,y,normalization='sequences',**kwargs)
            expected=torch.stack([function(logits[i:i+1],y[i:i+1],**kwargs) for i in range(3)]).mean()
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
            actual_grad=torch.autograd.grad(actual,logits,retain_graph=True)[0]
            expected_grad=torch.autograd.grad(expected,logits,retain_graph=True)[0]
            torch.testing.assert_close(actual_grad,expected_grad,atol=1e-7,rtol=1e-6)

    def test_sequence_gradient_accumulation_matches_combined_batch(self):
        torch.manual_seed(13)
        model=GPT(8,8,embedding_dim=8,num_heads=2,num_blocks=1,dropout=0)
        reference=deepcopy(model)
        x=torch.randint(0,8,(3,8));y=x.roll(-1,1);y[0,1:]=-100;y[1,4:]=-100
        base=partial(autoregressive_cross_entropy,normalization='sequences')
        expected=torch.optim.SGD(reference.parameters(),lr=.03)
        training_loss_with_unlikelihood(reference(x),y,base_loss_fn=base,normalization='sequences').backward()
        expected.step()
        with tempfile.TemporaryDirectory() as directory:
            config=TrainConfig(epochs=1,device='cpu',use_amp=False,loss_normalization='sequences',
                grad_accum_steps=3,grad_clip=0,warmup_steps=0,lr_decay_factor=1,checkpoint_every_steps=0,
                save_final_model_state=False,checkpoint_dir=directory)
            train_model(model,DataLoader(TensorDataset(x,y),batch_size=1),torch.optim.SGD(model.parameters(),lr=.03),base,config)
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)

    def test_cached_encoder_matches_native_bpe_and_rejects_only_over_limit(self):
        tok=Tokenizer();tok.vocab.update(' abcdef_123.,é')
        tok.token_to_id={t:i for i,t in enumerate(sorted(tok.vocab))}
        encoder=CachedEncoder(tok)
        for text in ('abc 123',' é., abc','abc\nabc','a_b'):
            native=tok.encode(text)
            self.assertEqual(encoder.encode(text,len(native)),native)
            if native:self.assertIsNone(encoder.encode(text,len(native)-1))
