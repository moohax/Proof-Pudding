# Proof Pudding (CVE-2019-20634)

This repository contains the code from our ([Will Pearce](https://twitter.com/moo_hax)/[Nick Landers](https://twitter.com/monoxgas)) 2019 DerbyCon presentation **"42: The answer to life, the universe, and everything offensive security"**. It is designed to attack ProofPoint's e-mail scoring system by stealing scored datasets (`core/data/*.csv`) and creating a copy-cat model for abuse. Before diving in, we'd recommend watching the **[presentation here](https://www.youtube.com/watch?v=CsvkYoxtexQ)**, or browse the **[slides here](https://github.com/moohax/Talks/blob/master/slides/DerbyCon19.pdf)**.

The project core is built on Python3 + Keras. It includes the stolen pre-scored datasets, pre-trained models (`./models/*`), and extracted insights (`./results/*`) from our research. It also exposes functionality for training, scoring, and reversing insights yourself.

## Training

Training is performed using an Artificial Neural Network (ANN) + Bag of Words tokenizing. We target the `mlxlogscore` for loss, which is generally a value between 1-999, with higher values representing "safer" samples. 

We've provided pre-trained models if you are reading this on a potato:

- `./models/texts.h5`
- `./models/links.h5`

To train your own model on link-based samples:
```
> python -m pip install -r requirements.txt
> python proofpudding.py train -d links ./models/my_link_model.h5

  ...
  
Epoch 8/10
10398/10398 [==============================] - 3s 298us/step - loss: 0.0428
Epoch 9/10
10398/10398 [==============================] - 3s 297us/step - loss: 0.0387
Epoch 10/10
10398/10398 [==============================] - 3s 295us/step - loss: 0.0369
2600/2600 [==============================] - 1s 271us/step

[+] Mean score error: 46

[+] Saved model to ./models/my_link_model.h5
[+] Saved vocab to ./models/my_link_model.h5.vocab
```

For text-based samples:
```
> python proofpudding.py train -d texts ./models/my_text_model.h5
```

The vocabulary for each model will be stored at `{model_file}.vocab`, and is required for performing scoring or insights. The performance of each model is measured in mean absolute error (MAE), which can effectively be converted into a "Mean Score Error", describing the mean average # of points we were off. The final measurement is taken from a split validation set of 20% by default.

To speed up training, we would recommend installing `tensorflow-gpu`.

## Scoring

With trained models, we can quickly score any sample to predict it's performance in the real world before delivery. Remember, you'll need to match your sample type (link or email) with a model which was trained on the correct data type (`-d`).

```
> python proofpudding.py score -m ./models/texts.h5 email.text
  ...
  
  [+] Predicted Score: 670


> python proofpudding.py score -m ./models/my_link_model.h5 http://link.com/file.txt
  ...
  
  [+] Predicted Score: 892
```

## Insights

During our research, we also created a basic approach to "reversing" the copy-cat model, attempting to list the highest and lowest scoring tokens. To do this, we take every sample and toggle any tokens which exist from `1` to `0`, then rescore the sample. We track the rolling score movement for each token, and divide it by the number of samples it appeared in. 

If you'd like to extract them yourself:
```
> python proofpudding.py insights -m ./models/my_link_model.h5 -d links ./results/my_results.csv
```

Using our pre-trained models, we've also pre-extracted insights for you:

- `./results/text_insights.csv`
- `./results/link_insights.csv`


**Here is a snippet of them:**

Good Link tokens: `category, title, song, depositphotos, archive`
*Bad* Link tokens: `wp, plugins, speadsheet, secret, battle, dispatch`

Good Text tokens: `gerald, thanks, fax, questions, blackberry`
*Bad* Text tokens: `gmt, image, home, payroll, xls, calendars, mr`

## Notes

- Proofpoint describe in their documentation that they have a model per client, so results for a law firm might be different than a hospital. However, it is unlikely Proofpoint trains a model for each client, ratherm, they probably fine-tune a larger, more general model.
- Each time the model is trained, results and insights will be slightly different - but should be similar across a number of training runs.
- There is some dead code that we'll revive at some point.

## Credits
Nancy Fulda, BYU
...
