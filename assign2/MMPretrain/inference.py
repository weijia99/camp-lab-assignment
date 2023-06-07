from mmpretrain import ImageClassificationInferencer

inference = ImageClassificationInferencer('res50.py','exp/best_accuracy_top1_epoch_74.pth')
result = inference('demo/pineapple.jpg')
print(result[0]['pred_class'])