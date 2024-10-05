Using device: cuda
Training model with progressive domain adaptation
Number of epochs: 100
Patience: 10
Domains: ['art_painting', 'cartoon', 'photo', 'sketch']
Device: cuda
Number of training samples: 6993
Number of validation samples: 1499
Number of test samples: 1499
Epoch 1/100
Training: 100%|██████████| 219/219 [02:05<00:00,  1.74it/s]
Epoch 1, Loss: 2.1184, Recon: 2.4993, Clf: 2.3186, KL: 0.0286
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:23<00:00,  2.04it/s]
Validation Fine-tuning Loss on art_painting: 1.8021
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.00it/s]
Validation Fine-tuning Loss on cartoon: 1.7791
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.00it/s]
Validation Fine-tuning Loss on photo: 1.7396
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  2.95it/s]
Validation Fine-tuning Loss on sketch: 1.7205
Epoch 2/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 2, Loss: 1.7024, Recon: 2.0875, Clf: 1.8505, KL: 0.1326
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.95it/s]
Validation Fine-tuning Loss on art_painting: 1.4914
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.00it/s]
Validation Fine-tuning Loss on cartoon: 1.4539
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.28it/s]
Validation Fine-tuning Loss on photo: 1.4381
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.09it/s]
Validation Fine-tuning Loss on sketch: 1.4359
Epoch 3/100
Training: 100%|██████████| 219/219 [01:29<00:00,  2.44it/s]
Epoch 3, Loss: 1.5378, Recon: 1.9711, Clf: 1.6360, KL: 0.3187
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.99it/s]
Validation Fine-tuning Loss on art_painting: 1.3621
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on cartoon: 1.3374
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.24it/s]
Validation Fine-tuning Loss on photo: 1.3458
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Validation Fine-tuning Loss on sketch: 1.3187
Epoch 4/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.41it/s]
Epoch 4, Loss: 1.4896, Recon: 1.9173, Clf: 1.5684, KL: 0.4317
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.95it/s]
Validation Fine-tuning Loss on art_painting: 1.2824
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.12it/s]
Validation Fine-tuning Loss on cartoon: 1.2723
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.24it/s]
Validation Fine-tuning Loss on photo: 1.2815
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Validation Fine-tuning Loss on sketch: 1.2774
Epoch 5/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 5, Loss: 1.4728, Recon: 1.8790, Clf: 1.5439, KL: 0.4977
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.00it/s]
Validation Fine-tuning Loss on art_painting: 1.2408
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on cartoon: 1.2258
Fine-tuning on photo: 100%|██████████| 47/47 [00:13<00:00,  3.37it/s]
Validation Fine-tuning Loss on photo: 1.2401
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.10it/s]
Validation Fine-tuning Loss on sketch: 1.2492
--- Evaluating on Test Set at Epoch 5 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:23<00:00,  1.99it/s]
Test Loss on art_painting: 1.2167, Test Accuracy on art_painting: 65.04%
Evaluating on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.33it/s]
Test Loss on cartoon: 1.2184, Test Accuracy on cartoon: 63.64%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Test Loss on photo: 1.2416, Test Accuracy on photo: 65.04%
Evaluating on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Test Loss on sketch: 1.2485, Test Accuracy on sketch: 65.04%
Epoch 6/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.42it/s]
Epoch 6, Loss: 1.4545, Recon: 1.8449, Clf: 1.5206, KL: 0.5356
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.02it/s]
Validation Fine-tuning Loss on art_painting: 1.2323
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Validation Fine-tuning Loss on cartoon: 1.2100
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on photo: 1.2190
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.09it/s]
Validation Fine-tuning Loss on sketch: 1.2094
Epoch 7/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 7, Loss: 1.4482, Recon: 1.8349, Clf: 1.5106, KL: 0.5623
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Validation Fine-tuning Loss on art_painting: 1.2177
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.28it/s]
Validation Fine-tuning Loss on cartoon: 1.1897
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on photo: 1.2137
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on sketch: 1.2038
Epoch 8/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.43it/s]
Epoch 8, Loss: 1.4393, Recon: 1.7794, Clf: 1.5049, KL: 0.5744
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.07it/s]
Validation Fine-tuning Loss on art_painting: 1.1864
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on cartoon: 1.1664
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on photo: 1.1928
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on sketch: 1.1757
Epoch 9/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.38it/s]
Epoch 9, Loss: 1.4236, Recon: 1.7757, Clf: 1.4853, KL: 0.5773
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.01it/s]
Validation Fine-tuning Loss on art_painting: 1.1854
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on cartoon: 1.1539
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on photo: 1.1897
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.34it/s]
Validation Fine-tuning Loss on sketch: 1.1781
Epoch 10/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.42it/s]
Epoch 10, Loss: 1.4463, Recon: 1.7154, Clf: 1.5213, KL: 0.5773
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.04it/s]
Validation Fine-tuning Loss on art_painting: 1.1791
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on cartoon: 1.1613
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on photo: 1.1636
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Validation Fine-tuning Loss on sketch: 1.1716
--- Evaluating on Test Set at Epoch 10 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Test Loss on art_painting: 1.1773, Test Accuracy on art_painting: 68.71%
Evaluating on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Test Loss on cartoon: 1.1627, Test Accuracy on cartoon: 68.51%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.30it/s]
Test Loss on photo: 1.1990, Test Accuracy on photo: 67.31%
Evaluating on sketch: 100%|██████████| 47/47 [00:13<00:00,  3.36it/s]
Test Loss on sketch: 1.1910, Test Accuracy on sketch: 66.78%
Epoch 11/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 11, Loss: 1.4138, Recon: 1.7254, Clf: 1.4799, KL: 0.5742
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Validation Fine-tuning Loss on art_painting: 1.1736
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Validation Fine-tuning Loss on cartoon: 1.1255
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on photo: 1.1728
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.28it/s]
Validation Fine-tuning Loss on sketch: 1.1583
Epoch 12/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.38it/s]
Epoch 12, Loss: 1.4011, Recon: 1.7652, Clf: 1.4597, KL: 0.5681
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.96it/s]
Validation Fine-tuning Loss on art_painting: 1.1508
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.02it/s]
Validation Fine-tuning Loss on cartoon: 1.1283
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on photo: 1.1498
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on sketch: 1.1503
Epoch 13/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 13, Loss: 1.4202, Recon: 1.6969, Clf: 1.4924, KL: 0.5662
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.10it/s]
Validation Fine-tuning Loss on art_painting: 1.1274
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on cartoon: 1.1319
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.24it/s]
Validation Fine-tuning Loss on photo: 1.1512
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.07it/s]
Validation Fine-tuning Loss on sketch: 1.1345
Epoch 14/100
Training: 100%|██████████| 219/219 [01:29<00:00,  2.44it/s]
Epoch 14, Loss: 1.4181, Recon: 1.7080, Clf: 1.4893, KL: 0.5588
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.17it/s]
Validation Fine-tuning Loss on art_painting: 1.1276
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on cartoon: 1.1238
Fine-tuning on photo: 100%|██████████| 47/47 [00:13<00:00,  3.38it/s]
Validation Fine-tuning Loss on photo: 1.1314
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.24it/s]
Validation Fine-tuning Loss on sketch: 1.1410
Epoch 15/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.35it/s]
Epoch 15, Loss: 1.3963, Recon: 1.6957, Clf: 1.4637, KL: 0.5571
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.12it/s]
Validation Fine-tuning Loss on art_painting: 1.1269
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on cartoon: 1.1234
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.25it/s]
Validation Fine-tuning Loss on photo: 1.1141
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on sketch: 1.1282
--- Evaluating on Test Set at Epoch 15 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.25it/s]
Test Loss on art_painting: 1.1338, Test Accuracy on art_painting: 70.05%
Evaluating on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.49it/s]
Test Loss on cartoon: 1.1240, Test Accuracy on cartoon: 71.11%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Test Loss on photo: 1.1712, Test Accuracy on photo: 68.45%
Evaluating on sketch: 100%|██████████| 47/47 [00:13<00:00,  3.36it/s]
Test Loss on sketch: 1.1410, Test Accuracy on sketch: 70.45%
Epoch 16/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 16, Loss: 1.3751, Recon: 1.6901, Clf: 1.4392, KL: 0.5473
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:16<00:00,  2.90it/s]
Validation Fine-tuning Loss on art_painting: 1.1277
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on cartoon: 1.1191
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.10it/s]
Validation Fine-tuning Loss on photo: 1.1062
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on sketch: 1.1196
Epoch 17/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 17, Loss: 1.4126, Recon: 1.6430, Clf: 1.4935, KL: 0.5347
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on art_painting: 1.1137
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on cartoon: 1.1110
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.29it/s]
Validation Fine-tuning Loss on photo: 1.1078
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.10it/s]
Validation Fine-tuning Loss on sketch: 1.1215
Epoch 18/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.41it/s]
Epoch 18, Loss: 1.3854, Recon: 1.6631, Clf: 1.4565, KL: 0.5388
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on art_painting: 1.1069
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on cartoon: 1.0924
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on photo: 1.1003
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on sketch: 1.0950
Epoch 19/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.42it/s]
Epoch 19, Loss: 1.3953, Recon: 1.6312, Clf: 1.4741, KL: 0.5292
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.10it/s]
Validation Fine-tuning Loss on art_painting: 1.1258
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.39it/s]
Validation Fine-tuning Loss on cartoon: 1.1067
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.12it/s]
Validation Fine-tuning Loss on photo: 1.1170
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on sketch: 1.1098
Epoch 20/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 20, Loss: 1.3911, Recon: 1.6115, Clf: 1.4717, KL: 0.5261
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.03it/s]
Validation Fine-tuning Loss on art_painting: 1.1225
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on cartoon: 1.1030
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.28it/s]
Validation Fine-tuning Loss on photo: 1.1072
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.34it/s]
Validation Fine-tuning Loss on sketch: 1.1009
--- Evaluating on Test Set at Epoch 20 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.13it/s]
Test Loss on art_painting: 1.1279, Test Accuracy on art_painting: 70.91%
Evaluating on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.33it/s]
Test Loss on cartoon: 1.1281, Test Accuracy on cartoon: 69.78%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.29it/s]
Test Loss on photo: 1.1509, Test Accuracy on photo: 68.58%
Evaluating on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Test Loss on sketch: 1.1472, Test Accuracy on sketch: 69.91%
Epoch 21/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 21, Loss: 1.4035, Recon: 1.5910, Clf: 1.4893, KL: 0.5293
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Validation Fine-tuning Loss on art_painting: 1.1236
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.34it/s]
Validation Fine-tuning Loss on cartoon: 1.1014
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.12it/s]
Validation Fine-tuning Loss on photo: 1.1042
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.02it/s]
Validation Fine-tuning Loss on sketch: 1.1097
Epoch 22/100
Training: 100%|██████████| 219/219 [01:29<00:00,  2.43it/s]
Epoch 22, Loss: 1.3734, Recon: 1.5880, Clf: 1.4523, KL: 0.5273
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.95it/s]
Validation Fine-tuning Loss on art_painting: 1.0942
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.35it/s]
Validation Fine-tuning Loss on cartoon: 1.0836
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on photo: 1.0981
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Validation Fine-tuning Loss on sketch: 1.0833
Epoch 23/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.36it/s]
Epoch 23, Loss: 1.4050, Recon: 1.5652, Clf: 1.4959, KL: 0.5174
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.05it/s]
Validation Fine-tuning Loss on art_painting: 1.1152
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on cartoon: 1.1079
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Validation Fine-tuning Loss on photo: 1.1223
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on sketch: 1.1080
Epoch 24/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 24, Loss: 1.3823, Recon: 1.5696, Clf: 1.4669, KL: 0.5180
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.02it/s]
Validation Fine-tuning Loss on art_painting: 1.1149
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on cartoon: 1.0951
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on photo: 1.1084
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.04it/s]
Validation Fine-tuning Loss on sketch: 1.1073
Epoch 25/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 25, Loss: 1.3607, Recon: 1.5925, Clf: 1.4374, KL: 0.5152
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on art_painting: 1.0960
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on cartoon: 1.0868
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Validation Fine-tuning Loss on photo: 1.0896
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on sketch: 1.0922
--- Evaluating on Test Set at Epoch 25 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Test Loss on art_painting: 1.0889, Test Accuracy on art_painting: 73.05%
Evaluating on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.34it/s]
Test Loss on cartoon: 1.1165, Test Accuracy on cartoon: 70.65%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Test Loss on photo: 1.1181, Test Accuracy on photo: 70.45%
Evaluating on sketch: 100%|██████████| 47/47 [00:16<00:00,  2.91it/s]
Test Loss on sketch: 1.1230, Test Accuracy on sketch: 71.38%
Epoch 26/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 26, Loss: 1.3824, Recon: 1.5618, Clf: 1.4691, KL: 0.5100
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:16<00:00,  2.83it/s]
Validation Fine-tuning Loss on art_painting: 1.0934
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on cartoon: 1.0749
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on photo: 1.0901
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.24it/s]
Validation Fine-tuning Loss on sketch: 1.0812
Epoch 27/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 27, Loss: 1.3695, Recon: 1.5776, Clf: 1.4515, KL: 0.5055
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.00it/s]
Validation Fine-tuning Loss on art_painting: 1.0972
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.33it/s]
Validation Fine-tuning Loss on cartoon: 1.0780
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Validation Fine-tuning Loss on photo: 1.0925
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on sketch: 1.0764
Epoch 28/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 28, Loss: 1.3962, Recon: 1.5153, Clf: 1.4925, KL: 0.5067
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.09it/s]
Validation Fine-tuning Loss on art_painting: 1.0955
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.24it/s]
Validation Fine-tuning Loss on cartoon: 1.0809
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on photo: 1.0965
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on sketch: 1.0764
Epoch 29/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 29, Loss: 1.3680, Recon: 1.5689, Clf: 1.4514, KL: 0.4998
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.12it/s]
Validation Fine-tuning Loss on art_painting: 1.0899
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.14it/s]
Validation Fine-tuning Loss on cartoon: 1.0822
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Validation Fine-tuning Loss on photo: 1.0731
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Validation Fine-tuning Loss on sketch: 1.0893
Epoch 30/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.41it/s]
Epoch 30, Loss: 1.3781, Recon: 1.5322, Clf: 1.4689, KL: 0.4978
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.97it/s]
Validation Fine-tuning Loss on art_painting: 1.0862
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.40it/s]
Validation Fine-tuning Loss on cartoon: 1.0837
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on photo: 1.0718
Fine-tuning on sketch: 100%|██████████| 47/47 [00:13<00:00,  3.39it/s]
Validation Fine-tuning Loss on sketch: 1.0664
--- Evaluating on Test Set at Epoch 30 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Test Loss on art_painting: 1.1043, Test Accuracy on art_painting: 71.91%
Evaluating on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.50it/s]
Test Loss on cartoon: 1.0997, Test Accuracy on cartoon: 72.38%
Evaluating on photo: 100%|██████████| 47/47 [00:13<00:00,  3.41it/s]
Test Loss on photo: 1.1104, Test Accuracy on photo: 72.25%
Evaluating on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Test Loss on sketch: 1.1154, Test Accuracy on sketch: 70.45%
Epoch 31/100
Training: 100%|██████████| 219/219 [01:29<00:00,  2.44it/s]
Epoch 31, Loss: 1.3735, Recon: 1.5269, Clf: 1.4641, KL: 0.4947
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on art_painting: 1.0873
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on cartoon: 1.0650
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Validation Fine-tuning Loss on photo: 1.0803
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on sketch: 1.0717
Epoch 32/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.42it/s]
Epoch 32, Loss: 1.3765, Recon: 1.5281, Clf: 1.4676, KL: 0.4967
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.01it/s]
Validation Fine-tuning Loss on art_painting: 1.0969
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Validation Fine-tuning Loss on cartoon: 1.0682
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on photo: 1.0804
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on sketch: 1.0725
Epoch 33/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 33, Loss: 1.3646, Recon: 1.5200, Clf: 1.4539, KL: 0.4952
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.13it/s]
Validation Fine-tuning Loss on art_painting: 1.0641
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on cartoon: 1.0585
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.17it/s]
Validation Fine-tuning Loss on photo: 1.0679
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.32it/s]
Validation Fine-tuning Loss on sketch: 1.0666
Epoch 34/100
Training: 100%|██████████| 219/219 [01:29<00:00,  2.45it/s]
Epoch 34, Loss: 1.3783, Recon: 1.5188, Clf: 1.4708, KL: 0.4976
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.06it/s]
Validation Fine-tuning Loss on art_painting: 1.0828
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.32it/s]
Validation Fine-tuning Loss on cartoon: 1.0644
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on photo: 1.0816
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.25it/s]
Validation Fine-tuning Loss on sketch: 1.0650
Epoch 35/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 35, Loss: 1.3550, Recon: 1.5125, Clf: 1.4422, KL: 0.4999
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Validation Fine-tuning Loss on art_painting: 1.0760
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.17it/s]
Validation Fine-tuning Loss on cartoon: 1.0704
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Validation Fine-tuning Loss on photo: 1.0783
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.25it/s]
Validation Fine-tuning Loss on sketch: 1.0673
--- Evaluating on Test Set at Epoch 35 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Test Loss on art_painting: 1.0938, Test Accuracy on art_painting: 71.65%
Evaluating on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.36it/s]
Test Loss on cartoon: 1.0984, Test Accuracy on cartoon: 71.78%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.20it/s]
Test Loss on photo: 1.1125, Test Accuracy on photo: 70.25%
Evaluating on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Test Loss on sketch: 1.0949, Test Accuracy on sketch: 72.78%
Epoch 36/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.42it/s]
Epoch 36, Loss: 1.3643, Recon: 1.5276, Clf: 1.4529, KL: 0.4926
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.04it/s]
Validation Fine-tuning Loss on art_painting: 1.0833
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on cartoon: 1.0705
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on photo: 1.0817
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.35it/s]
Validation Fine-tuning Loss on sketch: 1.0677
Epoch 37/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 37, Loss: 1.3717, Recon: 1.4691, Clf: 1.4704, KL: 0.4853
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Validation Fine-tuning Loss on art_painting: 1.0807
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on cartoon: 1.0757
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.07it/s]
Validation Fine-tuning Loss on photo: 1.0623
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.01it/s]
Validation Fine-tuning Loss on sketch: 1.0393
Epoch 38/100
Training: 100%|██████████| 219/219 [01:33<00:00,  2.34it/s]
Epoch 38, Loss: 1.3947, Recon: 1.4692, Clf: 1.4991, KL: 0.4847
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.96it/s]
Validation Fine-tuning Loss on art_painting: 1.0812
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.05it/s]
Validation Fine-tuning Loss on cartoon: 1.0674
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.10it/s]
Validation Fine-tuning Loss on photo: 1.0852
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.30it/s]
Validation Fine-tuning Loss on sketch: 1.0667
Epoch 39/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.42it/s]
Epoch 39, Loss: 1.3711, Recon: 1.4659, Clf: 1.4698, KL: 0.4869
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.25it/s]
Validation Fine-tuning Loss on art_painting: 1.0713
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.39it/s]
Validation Fine-tuning Loss on cartoon: 1.0581
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.33it/s]
Validation Fine-tuning Loss on photo: 1.0591
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on sketch: 1.0592
Epoch 40/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.43it/s]
Epoch 40, Loss: 1.3642, Recon: 1.4830, Clf: 1.4587, KL: 0.4887
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on art_painting: 1.0736
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on cartoon: 1.0533
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on photo: 1.0425
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on sketch: 1.0655
--- Evaluating on Test Set at Epoch 40 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Test Loss on art_painting: 1.0928, Test Accuracy on art_painting: 71.65%
Evaluating on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.41it/s]
Test Loss on cartoon: 1.0851, Test Accuracy on cartoon: 72.78%
Evaluating on photo: 100%|██████████| 47/47 [00:13<00:00,  3.50it/s]
Test Loss on photo: 1.0845, Test Accuracy on photo: 72.85%
Evaluating on sketch: 100%|██████████| 47/47 [00:13<00:00,  3.51it/s]
Test Loss on sketch: 1.0790, Test Accuracy on sketch: 74.05%
Epoch 41/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.39it/s]
Epoch 41, Loss: 1.3778, Recon: 1.4668, Clf: 1.4783, KL: 0.4842
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.00it/s]
Validation Fine-tuning Loss on art_painting: 1.0856
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.40it/s]
Validation Fine-tuning Loss on cartoon: 1.0661
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.17it/s]
Validation Fine-tuning Loss on photo: 1.0556
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on sketch: 1.0483
Epoch 42/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 42, Loss: 1.3571, Recon: 1.5070, Clf: 1.4470, KL: 0.4877
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on art_painting: 1.0716
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.21it/s]
Validation Fine-tuning Loss on cartoon: 1.0650
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.27it/s]
Validation Fine-tuning Loss on photo: 1.0691
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on sketch: 1.0723
Epoch 43/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.37it/s]
Epoch 43, Loss: 1.3702, Recon: 1.4668, Clf: 1.4682, KL: 0.4900
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.96it/s]
Validation Fine-tuning Loss on art_painting: 1.0625
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.07it/s]
Validation Fine-tuning Loss on cartoon: 1.0619
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on photo: 1.0489
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.07it/s]
Validation Fine-tuning Loss on sketch: 1.0472
Epoch 44/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 44, Loss: 1.3709, Recon: 1.4854, Clf: 1.4669, KL: 0.4884
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.02it/s]
Validation Fine-tuning Loss on art_painting: 1.0685
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.07it/s]
Validation Fine-tuning Loss on cartoon: 1.0546
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.14it/s]
Validation Fine-tuning Loss on photo: 1.0497
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.14it/s]
Validation Fine-tuning Loss on sketch: 1.0423
Epoch 45/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.41it/s]
Epoch 45, Loss: 1.3556, Recon: 1.4910, Clf: 1.4469, KL: 0.4903
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:16<00:00,  2.93it/s]
Validation Fine-tuning Loss on art_painting: 1.0617
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.29it/s]
Validation Fine-tuning Loss on cartoon: 1.0604
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Validation Fine-tuning Loss on photo: 1.0364
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.18it/s]
Validation Fine-tuning Loss on sketch: 1.0431
--- Evaluating on Test Set at Epoch 45 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.19it/s]
Test Loss on art_painting: 1.0961, Test Accuracy on art_painting: 71.45%
Evaluating on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.44it/s]
Test Loss on cartoon: 1.0824, Test Accuracy on cartoon: 72.72%
Evaluating on photo: 100%|██████████| 47/47 [00:13<00:00,  3.37it/s]
Test Loss on photo: 1.1006, Test Accuracy on photo: 72.78%
Evaluating on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.30it/s]
Test Loss on sketch: 1.0915, Test Accuracy on sketch: 72.65%
Epoch 46/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 46, Loss: 1.3636, Recon: 1.4861, Clf: 1.4581, KL: 0.4855
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.14it/s]
Validation Fine-tuning Loss on art_painting: 1.0563
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on cartoon: 1.0583
Fine-tuning on photo: 100%|██████████| 47/47 [00:15<00:00,  3.13it/s]
Validation Fine-tuning Loss on photo: 1.0384
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Validation Fine-tuning Loss on sketch: 1.0409
Epoch 47/100
Training: 100%|██████████| 219/219 [01:30<00:00,  2.41it/s]
Epoch 47, Loss: 1.3568, Recon: 1.4662, Clf: 1.4513, KL: 0.4909
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on art_painting: 1.0642
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.15it/s]
Validation Fine-tuning Loss on cartoon: 1.0405
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Validation Fine-tuning Loss on photo: 1.0349
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on sketch: 1.0355
Epoch 48/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 48, Loss: 1.3526, Recon: 1.4681, Clf: 1.4466, KL: 0.4858
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:16<00:00,  2.92it/s]
Validation Fine-tuning Loss on art_painting: 1.0487
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.31it/s]
Validation Fine-tuning Loss on cartoon: 1.0564
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on photo: 1.0489
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.14it/s]
Validation Fine-tuning Loss on sketch: 1.0415
Epoch 49/100
Training: 100%|██████████| 219/219 [01:32<00:00,  2.38it/s]
Epoch 49, Loss: 1.3698, Recon: 1.4463, Clf: 1.4700, KL: 0.4920
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:14<00:00,  3.17it/s]
Validation Fine-tuning Loss on art_painting: 1.0822
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:14<00:00,  3.26it/s]
Validation Fine-tuning Loss on cartoon: 1.0660
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.16it/s]
Validation Fine-tuning Loss on photo: 1.0630
Fine-tuning on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.22it/s]
Validation Fine-tuning Loss on sketch: 1.0601
Epoch 50/100
Training: 100%|██████████| 219/219 [01:31<00:00,  2.40it/s]
Epoch 50, Loss: 1.3488, Recon: 1.4562, Clf: 1.4430, KL: 0.4881
Weights - Alpha: 0.1000, Beta: 0.8000, Gamma: 0.1000
Fine-tuning on art_painting: 100%|██████████| 47/47 [00:15<00:00,  2.94it/s]
Validation Fine-tuning Loss on art_painting: 1.0473
Fine-tuning on cartoon: 100%|██████████| 47/47 [00:15<00:00,  3.11it/s]
Validation Fine-tuning Loss on cartoon: 1.0462
Fine-tuning on photo: 100%|██████████| 47/47 [00:14<00:00,  3.23it/s]
Validation Fine-tuning Loss on photo: 1.0352
Fine-tuning on sketch: 100%|██████████| 47/47 [00:15<00:00,  3.08it/s]
Validation Fine-tuning Loss on sketch: 1.0396
--- Evaluating on Test Set at Epoch 50 ---
Evaluating on art_painting: 100%|██████████| 47/47 [00:15<00:00,  3.01it/s]
Test Loss on art_painting: 1.0632, Test Accuracy on art_painting: 74.52%
Evaluating on cartoon: 100%|██████████| 47/47 [00:13<00:00,  3.42it/s]
Test Loss on cartoon: 1.0841, Test Accuracy on cartoon: 73.32%
Evaluating on photo: 100%|██████████| 47/47 [00:14<00:00,  3.35it/s]
Test Loss on photo: 1.0705, Test Accuracy on photo: 73.45%
Evaluating on sketch: 100%|██████████| 47/47 [00:14<00:00,  3.30it/s]
Test Loss on sketch: 1.0820, Test Accuracy on sketch: 71.85%
Epoch 51/100