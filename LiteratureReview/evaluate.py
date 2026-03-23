from utilities import get_property_set
import os


all = [i.split(".")[0] for i in os.listdir("json/pah101")]
optical_gap_exp_true = set(["BENZEN", "ANTCEN", "TETCEN01", "PENCEN", "ZZZDKE01", "QQQCIG04", "QQQCIG13", "QQQCIG14", "PERLEN05", "PERLEN07", "POBPIG", "QUATER10", "CORONE01", "HBZCOR", "BEANTR", "BIPHEN", "CRYSEN01", "TERPHE02", "BNPERY", "KUBVUY", "KUBWAF01"])
optical_gap_cal_true = set(["BENZEN", "ANTCEN", "TETCEN01", "PENCEN", "ZZZDKE01"])
absorption_true = set(["BEANTR", "BENZEN", "BIPHEN", "BNPERY", "CORANN12", "CORONE01", "CRYSEN01", "DPANTR01", "HBZCOR", "KUBVUY", "KUBWAF01", "TERPHE02", "TETCEN01", "TRIPHE12", "ZZZOYC01"])

print(len(all), "structures in the database.")
print()



optical_gap_exp_positive, optical_gap_exp_neutral, _, _ = get_property_set("LiteratureReview/pah101_nano.json", "llm_exp", ["optical_gap", "optical_gap_source"])
absorption_positive, absorption_neutral, _, _ = get_property_set("LiteratureReview/pah101_nano.json", "llm_exp", ["absorption_spectrum"])
print("optical_gap false negative:", optical_gap_exp_true - optical_gap_exp_positive)
print("absorption false negative:", absorption_true - absorption_positive)

def print_confusion_matrix(true_set: set, pred_set: set, label):
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = len(set(all) - (true_set | pred_set))
    print(f"{label} confusion matrix:")
    print(f"  True Positive:  {tp}")
    print(f"  False Positive: {fp}")
    print(f"  False Negative: {fn}")
    print(f"  True Negative:  {tn}")
    # Calculate accuracy, precision, recall, F1
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"  Accuracy:       {accuracy:.3f}")
    print(f"  Precision:      {precision:.3f}")
    print(f"  Recall:         {recall:.3f}")
    print(f"  F1 Score:       {f1:.3f}")
    print()


print_confusion_matrix(optical_gap_exp_true, optical_gap_exp_positive, "Optical Gap (Exp)")
print_confusion_matrix(optical_gap_exp_true, optical_gap_exp_positive | optical_gap_exp_neutral, "Optical Gap (Exp) Neutral")
print_confusion_matrix(absorption_true, absorption_positive, "Absorption")
print_confusion_matrix(absorption_true, absorption_positive | absorption_neutral, "Absorption Neutral")


"""
papers cannot be found by human: 2
papers cannot be found by scraping: 2
I did not find proper experiments: 2
proper experiments not found by LLM: 2
"""