Related work

ORBIS autnomous driving world model
Diffusion models
Overfitting
Finetuning on world model latents
NAVSIM planning 


World model p(s', r|s, a)
Policy p(a|s)

Planning in inference: sample actions, get rewards from world model, select best action sequence

## What's the difference between world model and video generator?

No fundamental difference. In practice:
- world model is usually conditioned on actions
- usually predict in latent space
- judged by how well the planning works (but no loss enforcing this!)

The JEPA debate: is pixel-level reconstruction a good enough target to gain semantic understanding?

---

Introduction

Unsupervised learning. Unsupervised prediction of missing parts of the video, analogous to next-token prediction in text, is the strongest learning signal to enable general scene understanding. As we know from Richard Sutton's bitter lesson (footnote: http://www.incompleteideas.net/IncIdeas/BitterLesson.html), in the long run, the best-performing methods are based on increasing the amount of computation and removing expert-engineered biases. Similarly, in the famous Yann LeCun's cake analogy (footnote: https://medium.com/syncedreview/yann-lecun-cake-analogy-2-0-a361da560dae), the biggest learning signal comes from unsupervised pretraining. Manual engineering is still relevant for small data, rare tasks and for performance reasons.

Video world models. Prior to learned video world models, the foundational works in unsupervised learning such as SimCLR, CLIP, and DINO leveraged big amounts of unlabeled images. The next step in learning without annotations were transformer-based LLMs. They may be called **language world models**, but they are by design different from video world models:
1. Text is an ambiguous, indirect observation of reality.
2. No action input

Video generators and world models differ in their application. Video generators 




and  Ca

Implicit motion estimation.
