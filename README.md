
# Simple Learning to Optimize in PyTorch

Authors: Boris Knyazev, Abhinav Moudgil, Guillaume Lajoie, Eugene Belilovsky

This code is based on the methods mainly from:

1. Learning to learn by gradient descent by gradient descent https://arxiv.org/abs/1606.04474
2. Understanding and correcting pathologies in the training of learned optimizers https://arxiv.org/abs/1810.10180
3. Code from https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/no_dependency_learned_optimizer.ipynb

# Requirements

- Python 3.6+
- PyTorch 1.9+

# Usage

## Training

    python l2o_train.py

By default, it will be meta-trained on the `FashionMNIST` task (`-t 0`).
The checkpoints will be saved as the `results/<path_to_model>/step_XXX.pt` files,
where `<path_to_model>` is generated based on the command line arguments.
The best checkpoint will be saved as `results/<path_to_model>/best.pt`.

## Testing

    python l2o_test.py --ckpt results/<path_to_model>/best.pt -t <task>

where `task` 0 by default (see `tasks.py` for the list of tasks).

# License

This code is licensed under [MIT license](LICENSE).

# Citation

If you use this code in your research, please cite it as the following:
```
@article{knyazev2023l2opytorch,
  title={Simple Learning to Optimize in PyTorch},
  author={Knyazev, Boris and Moudgil, Abhinav and Lajoie, Guillaume and Belilovsky, Eugene},
  source={GitHub: https://github.com/SamsungSAILMontreal/l2o_pytorch},
  year={2023}
}
```