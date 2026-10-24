import unittest
import torch
from src.train import Config, LABELS, NewsMLP, preprocess_text, load_official_data

class TestCheckpoint1(unittest.TestCase):
    def test_clean_html_and_negation(self):
        self.assertEqual(preprocess_text('Not <b>good</b> &amp; bad!'), 'not good bad')
    def test_model_shapes_and_backward(self):
        model = NewsMLP(10, 8, 4, .2)
        logits = model(torch.randn(3, 10))
        self.assertEqual(tuple(logits.shape), (3, 4))
        loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0,1,3]))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
    def test_original_splits_are_separate(self):
        train, test = load_official_data(Config())
        self.assertEqual((len(train), len(test)), (8000,2000))
        self.assertEqual(set(train.label.unique()), set(LABELS))
        self.assertEqual(set(test.label.unique()), set(LABELS))

if __name__ == '__main__':
    unittest.main()
