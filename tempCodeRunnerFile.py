print(f"AUC-ROC: {roc_auc_score(valY, clf.predict_proba(val_features), multi_class='ovr'):.4f}")
# print(f"MCC: {matthews_corrcoef(valY, preds):.4f}")